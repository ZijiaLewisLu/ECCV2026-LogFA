from __future__ import annotations

import json
import os
import random
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from yacs.config import CfgNode

from .utils import parse_label, shrink_frame_label


MODES = {
    "local_aug": "local_aug",
    "img_textprompt_interp2": "local_aug",
    "global_aug": "global_aug",
    "img_trans_gdag": "global_aug",
    "logfa": "logfa",
    "tprompt_interp2_trans_gdag": "logfa",
}


# Supported datasets and their default release-package directory name.
DATASET_PACKAGE = {
    "egoper": "LogFA-EgoPER-v1",
    "egoprocel": "LogFA-EgoProceL-v1",
}


def resolve_data_root(
    configured: str | os.PathLike | None = None, dataset: str = "egoper"
) -> Path:
    """Resolve the release-data root for the given dataset.

    Priority: explicit config/CLI, ``LOGFA_DATA_ROOT``, then ``./data``. The
    default package directory depends on the dataset (EgoPER / EgoProceL); the
    internal layout (``annotations/``, ``features/``, ``weights/``) is identical.
    """
    pkg = DATASET_PACKAGE.get(str(dataset).lower(), DATASET_PACKAGE["egoper"])
    value = configured or os.environ.get("LOGFA_DATA_ROOT")
    if value:
        candidates = [Path(value).expanduser().resolve()]
    else:
        repository = _source_root()
        candidates = [
            (Path.cwd() / "data" / pkg).resolve(),
            (repository / "data" / pkg).resolve(),
            (repository.parent / "data" / pkg).resolve(),
        ]

    for root in candidates:
        if (root / "annotations").is_dir():
            return root
        nested = root / pkg
        if (nested / "annotations").is_dir():
            return nested
    return candidates[0]


def _source_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _annotation_root(data_root: Path) -> Path:
    return data_root / "annotations"


def _feature_root(data_root: Path, name: str) -> Path:
    return data_root / "features" / name


def _load_float_array(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = np.load(path, allow_pickle=False)
    if isinstance(value, np.lib.npyio.NpzFile):
        value = value["data"]
    return np.asarray(value, dtype=np.float32)


def _load_mapping(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    label_to_index: dict[str, int] = {}
    index_to_label: dict[int, str] = {}
    with path.open() as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            index, label = line.split(maxsplit=1)
            label_to_index[label] = int(index)
            index_to_label[int(index)] = label
    return label_to_index, index_to_label


@dataclass(frozen=True)
class GDAGNode:
    id: str
    type: str
    actions: tuple[str, ...]
    c1: int
    c2: int


class GDAGPlanner:
    """Sample valid action sequences from a released GDAG definition.

    Consumes the ``{nodes, edges}`` GDAG format used by EgoPER (flat/hierarchical
    nodes with a repetition range ``c1..c2``). ``sample()`` returns a flat list of
    action names. Construct via :func:`load_task_graph`.
    """

    def __init__(self, graph: dict):
        self.nodes = {
            item["id"]: GDAGNode(
                id=item["id"],
                type=item["type"],
                actions=tuple(item.get("actions", ())),
                c1=int(item["c1"]),
                c2=int(item["c2"]),
            )
            for item in graph["nodes"]
        }
        self.successors = {node_id: [] for node_id in self.nodes}
        self.predecessors = {node_id: [] for node_id in self.nodes}
        for source, target in graph["edges"]:
            if source not in self.nodes or target not in self.nodes:
                raise ValueError(f"GDAG edge references unknown node: {source, target}")
            self.successors[source].append(target)
            self.predecessors[target].append(source)

    def sample(self) -> list[str]:
        indegree = {
            node_id: len(self.predecessors[node_id]) for node_id in self.nodes
        }
        ready = [node_id for node_id, degree in indegree.items() if degree == 0]
        sequence: list[str] = []

        while ready:
            node_id = random.choice(ready)
            ready.remove(node_id)
            node = self.nodes[node_id]

            if node.c1 == 0:
                repeat = 0 if random.random() < 0.5 else random.randint(
                    1, max(1, node.c2)
                )
            else:
                repeat = random.randint(node.c1, max(node.c1, node.c2))

            for _ in range(repeat):
                if node.type == "hierarchical":
                    sequence.extend(node.actions)
                else:
                    sequence.append(node.id)

            for target in self.successors[node_id]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)

        if any(degree > 0 for degree in indegree.values()):
            raise ValueError("Released GDAG contains a cycle")
        return sequence


def load_task_graph(path: Path):
    """Load a released GDAG definition (``{nodes, edges}``) as a GDAGPlanner.

    An LLM-response wrapper (``{"response": "...```json ...```"}``) is unwrapped
    first. Both EgoPER and EgoProceL use this ``{nodes, edges}`` format; for
    EgoProceL the graph is produced by
    ``augmentation/global_aug/generate_egoprocel_gdag.py``.
    """
    with Path(path).open() as file:
        data = json.load(file)
    if "response" in data and "nodes" not in data:
        response = data["response"]
        fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", response, re.DOTALL)
        data = json.loads(fence.group(1) if fence else response)
    if "nodes" in data and "edges" in data:
        return GDAGPlanner(data)
    raise ValueError(
        f"Unrecognized GDAG format in {path}: keys={sorted(data)}"
    )


def _transcript_shuffle(source, target_actions: Iterable[int], bg_label: int = 0):
    """Realize a sampled order by reusing observed source segments."""
    remaining = deepcopy(source)
    target_actions = list(target_actions)

    def first(action, segments):
        return next(
            (index for index, segment in enumerate(segments)
             if segment.action == action),
            -1,
        )

    for index, segment in enumerate(remaining):
        segment.idx = index
    backgrounds = [segment for segment in remaining if segment.action == bg_label]

    placed = []
    for action in target_actions:
        if action == bg_label:
            continue
        index = first(action, remaining)
        if index >= 0:
            segment = remaining.pop(index)
        else:
            index = first(action, placed)
            if index < 0:
                continue
            segment = placed[index]
        placed.append(segment)

    if not placed:
        return []

    positions = [-1, *[segment.idx for segment in placed]]
    positions.append(positions[-1] + 1)
    gaps = np.asarray([positions[:-1], positions[1:]], dtype=int).T
    original_gaps = np.asarray(
        [[segment.idx - 1, segment.idx + 1] for segment in backgrounds],
        dtype=int,
    ).reshape(-1, 2)

    placed_backgrounds = [None] * len(gaps)
    if len(original_gaps):
        matches = (gaps[:, None, :] == original_gaps[None, :, :]).sum(-1)
        used = set()
        exact_gap, exact_original = np.where(matches == 2)
        for gap_index, original_index in zip(exact_gap, exact_original):
            if original_index not in used:
                placed_backgrounds[gap_index] = backgrounds[original_index]
                used.add(int(original_index))
        for gap_index in range(len(gaps)):
            if placed_backgrounds[gap_index] is not None:
                continue
            candidates = [
                index for index in np.where(matches[gap_index] > 0)[0]
                if int(index) not in used
            ]
            if candidates:
                original_index = int(random.choice(candidates))
                placed_backgrounds[gap_index] = backgrounds[original_index]
                used.add(original_index)

    result = []
    for index, segment in enumerate(placed):
        if placed_backgrounds[index] is not None:
            result.append(placed_backgrounds[index])
        result.append(segment)
    if placed_backgrounds[-1] is not None:
        result.append(placed_backgrounds[-1])
    return result


class Dataset:
    """Lazy EgoPER video dataset using the public release-data layout."""

    def __init__(self, cfg: CfgNode, training: bool = True):
        self.dataset = str(cfg.dataset).lower()
        if self.dataset not in DATASET_PACKAGE:
            raise ValueError(
                f"Unsupported dataset={cfg.dataset!r}; "
                f"supported: {sorted(DATASET_PACKAGE)}"
            )
        if cfg.dname not in MODES:
            raise ValueError(f"Unsupported release mode: {cfg.dname}")
        if not cfg.recipe:
            raise ValueError("A task/recipe is required")

        self.cfg = cfg
        self.training = training
        self.mode = MODES[cfg.dname]
        self.data_root = resolve_data_root(
            getattr(cfg, "data_root", None), self.dataset
        )
        self.annotations = _annotation_root(self.data_root)
        self.base_features = _feature_root(self.data_root, "siglip2")
        # Local PFE is the prompt-config sweep set (18 configs per video);
        # cfg.tprompt_idx selects which one. There is no single collapsed set.
        self.tprompt_idx = int(getattr(cfg, "tprompt_idx", 0))
        configured_sweep_root = getattr(cfg, "local_pfe_sweep_root", None)
        self.local_pfe_sweep = (
            Path(configured_sweep_root).expanduser().resolve()
            if configured_sweep_root
            else _feature_root(self.data_root, "local_pfe_sweep")
        )

        self.label2index, self.index2label = _load_mapping(
            self.annotations / "mapping.txt"
        )
        self.nclasses = len(self.label2index)
        self.bg_class = [0]
        self.input_dimension = 1024

        split = cfg.split if training else (cfg.test_split or cfg.split)
        suffix = "train" if training else "test"
        split_path = (
            self.annotations / "splits" / cfg.recipe / f"{split}.{suffix}"
        )
        with split_path.open() as file:
            self.video_list = [line.strip() for line in file if line.strip()]

        self._cache = {}
        self.average_transcript_len = (
            self._compute_average_transcript_len() if training else None
        )
        self.gdag = None
        if self.mode in {"global_aug", "logfa"}:
            configured_graph_root = getattr(cfg, "gdag_root", None)
            graph_root = (
                Path(configured_graph_root).expanduser().resolve()
                if configured_graph_root
                else self.data_root / "gdag"
            )
            graph_path = graph_root / f"{cfg.recipe}.json"
            self.gdag = load_task_graph(graph_path)

    def __len__(self):
        return len(self.video_list)

    def __str__(self):
        return (
            f"<EgoPERDataset mode={self.mode} videos={len(self)} "
            f"features={self.input_dimension}>"
        )

    def get_vnames(self):
        return self.video_list[:]

    def __getitem__(self, video):
        """Public dataset item consumed by the model."""
        base, _local, _segments, labels, full_labels = self.get_augmented_item(
            video
        )
        return base, labels, full_labels

    def get_augmented_item(self, video):
        """Return private augmentation state consumed by this module's loader."""
        if video not in self.video_list:
            raise KeyError(video)
        if video not in self._cache:
            self._cache[video] = self._load_video(video)
        return self._cache[video]

    def _compute_average_transcript_len(self):
        lengths = []
        for video in self.video_list:
            with (
                self.annotations / "groundTruth" / f"{video}.txt"
            ).open() as file:
                labels = [
                    self.label2index[line.split("|", 1)[0]]
                    for line in file.read().splitlines()
                ]
            lengths.append(len(parse_label(shrink_frame_label(labels, 10))))
        return float(np.mean(lengths))

    @torch.no_grad()
    def _load_video(self, video):
        base = _load_float_array(self.base_features / f"{video}.npy")
        with (self.annotations / "groundTruth" / f"{video}.txt").open() as file:
            raw_labels = [line.split("|") for line in file.read().splitlines()]

        if self.cfg.error_vid:
            labels = []
            for fields in raw_labels:
                action = fields[0]
                error_type = fields[1] if len(fields) > 1 else "Normal"
                labels.append(
                    0 if error_type == "Error_Addition"
                    else self.label2index[action]
                )
        else:
            labels = [self.label2index[fields[0]] for fields in raw_labels]

        sample_rate = 10
        sampled_labels = shrink_frame_label(labels, sample_rate)
        base = base[::sample_rate]
        length = min(len(base), len(sampled_labels))
        base = base[:length]
        sampled_labels = sampled_labels[:length]

        local = None
        if self.training and self.mode in {"local_aug", "logfa"}:
            if self.tprompt_idx < 0:
                raise ValueError(
                    "tprompt_idx must be >= 0: the release ships only the "
                    "local_pfe_sweep set (0..17), not a single collapsed local_pfe."
                )
            # Prompt-config sweep: fix one prompt-learning config per run.
            local_path = (
                self.local_pfe_sweep
                / self.cfg.recipe
                / f"{video}_{self.tprompt_idx}.npy"
            )
            local = _load_float_array(local_path)
            if local.ndim == 2:
                local = local[:, None, :]
            if local.ndim != 3 or local.shape[-1] != base.shape[-1]:
                raise ValueError(
                    f"Expected local PFE [T,K,{base.shape[-1]}], got "
                    f"{local.shape} for {video}"
                )
            length = min(length, len(local))
            base = base[:length]
            local = local[:length]
            sampled_labels = sampled_labels[:length]

        segments = parse_label(sampled_labels)
        return base, local, segments, sampled_labels, labels


class DataLoader:
    """Small video-level batch loader compatible with the original trainer."""

    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.videos = dataset.get_vnames()
        self.selector = list(range(len(self.videos)))
        self.num_batch = int(np.ceil(len(self.videos) / batch_size))
        self.index = 0
        if shuffle:
            np.random.shuffle(self.selector)

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        return self

    def _apply_local_pfe(self, base: np.ndarray, local: np.ndarray) -> np.ndarray:
        choices = np.random.randint(local.shape[1], size=len(base))
        selected = local[np.arange(len(base)), choices]
        interpolation = np.random.random()
        return (
            (1.0 - interpolation) * base + interpolation * selected
        ).astype(np.float32)

    def _apply_gdag(self, sequence, segments, labels):
        target_names = self.dataset.gdag.sample()
        target = [
            self.dataset.label2index[name]
            for name in target_names
            if name != "BG" and name in self.dataset.label2index
        ]
        rearranged = _transcript_shuffle(segments, target, bg_label=0)
        if not rearranged:
            return sequence, labels

        parts = [sequence[segment.start:segment.end + 1] for segment in rearranged]
        new_labels = [
            segment.action
            for segment in rearranged
            for _ in range(segment.end - segment.start + 1)
        ]
        return np.concatenate(parts, axis=0), new_labels

    def __next__(self):
        if self.index >= len(self.videos):
            if self.shuffle:
                np.random.shuffle(self.selector)
            self.index = 0
            raise StopIteration

        indices = self.selector[self.index:self.index + self.batch_size]
        if len(indices) < self.batch_size:
            indices += self.selector[:self.batch_size - len(indices)]
        self.index += self.batch_size
        videos = [self.videos[index] for index in indices]

        sequences, train_labels, eval_labels = [], [], []
        # Unified taug_ratio (>= 0) overrides both per-branch probabilities.
        unified = float(getattr(self.dataset.cfg, "taug_ratio", -1.0))
        if unified >= 0.0:
            local_probability = unified
            global_probability = unified
        else:
            local_probability = float(
                getattr(
                    self.dataset.cfg,
                    "local_aug_prob",
                    0.0,
                )
            )
            global_probability = float(
                getattr(
                    self.dataset.cfg,
                    "global_aug_prob",
                    0.0,
                )
            )

        for video in videos:
            base, local, segments, labels, full_labels = (
                self.dataset.get_augmented_item(video)
            )
            sequence = base
            train_label = labels
            eval_label = full_labels

            if (
                self.dataset.training
                and self.dataset.mode in {"local_aug", "logfa"}
                and np.random.random() < local_probability
            ):
                sequence = self._apply_local_pfe(sequence, local)

            if (
                self.dataset.training
                and self.dataset.mode in {"global_aug", "logfa"}
                and np.random.random() < global_probability
            ):
                sequence, train_label = self._apply_gdag(
                    sequence, segments, train_label
                )
                eval_label = train_label

            sequences.append(torch.from_numpy(sequence).float())
            train_labels.append(torch.as_tensor(train_label, dtype=torch.long))
            eval_labels.append(eval_label)

        return videos, sequences, train_labels, eval_labels


def create_dataset(cfg: CfgNode):
    test_dataset = Dataset(cfg, training=False)
    train_dataset = Dataset(cfg, training=True)
    return train_dataset, test_dataset
