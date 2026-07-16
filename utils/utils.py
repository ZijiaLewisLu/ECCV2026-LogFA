from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np


def file_sha256(path, chunk_size=1024 * 1024):
    """SHA256 of a file, streamed in chunks."""
    digest = sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Segment():
    def __init__(self, action, start, end):
        assert start >= 0
        self.action = action
        self.start = start
        self.end = end
        self.len = end - start + 1

    def __repr__(self):
        return "<%r %d-%d>" % (self.action, self.start, self.end)

    def intersect(self, s2):
        s = max([self.start, s2.start])
        e = min([self.end, s2.end])
        return max(0, e - s + 1)

    def union(self, s2):
        s = min([self.start, s2.start])
        e = max([self.end, s2.end])
        return e - s + 1


def parse_label(label: np.array):
    """Convert frame-wise labels into a list of ``Segment`` objects."""
    if not isinstance(label, np.ndarray):
        label = np.array(label)

    loc = label[:-1] != label[1:]
    loc = np.where(loc)[0]
    segs = []

    if len(loc) == 0:
        return [Segment(label[0].item(), 0, len(label) - 1)]

    for i, l in enumerate(loc):
        if i == 0:
            start = 0
            end = l
        else:
            start = loc[i - 1] + 1
            end = l

        seg = Segment(label[start].item(), start, end)
        segs.append(seg)

    segs.append(Segment(label[loc[-1] + 1].item(), loc[-1] + 1, len(label) - 1))
    return segs


def expand_frame_label(label, target_len: int):
    if len(label) == target_len:
        return label

    import torch
    is_numpy = isinstance(label, np.ndarray)
    if is_numpy:
        label = torch.from_numpy(label).float()
    if isinstance(label, list):
        label = torch.FloatTensor(label)

    label = label.view([1, 1, -1])
    resized = torch.nn.functional.interpolate(
        label, size=target_len, mode="nearest"
    ).view(-1)
    resized = resized.long()

    if is_numpy:
        resized = resized.detach().numpy()

    return resized


def shrink_frame_label(label: list, clip_len: int) -> list:
    num_clip = ((len(label) - 1) // clip_len) + 1
    new_label = []
    for i in range(num_clip):
        s = i * clip_len
        e = s + clip_len
        l = label[s:e]
        ct = Counter(l)
        l = ct.most_common()[0][0]
        new_label.append(l)

    return new_label


def easy_reduce(scores, mode="mean", skip_nan=False):
    assert isinstance(scores, list), type(scores)

    if len(scores) == 0:
        return np.nan

    elif isinstance(scores[0], list):
        average = []
        L = len(scores[0])
        for i in range(L):
            average.append(easy_reduce([s[i] for s in scores], mode=mode, skip_nan=skip_nan))

    elif isinstance(scores[0], np.ndarray):
        assert len(scores[0].shape) == 1
        stack = np.stack(scores, axis=0)
        average = stack.mean(0)

    elif isinstance(scores[0], tuple):
        average = []
        L = len(scores[0])
        for i in range(L):
            average.append(easy_reduce([s[i] for s in scores], mode=mode, skip_nan=skip_nan))
        average = tuple(average)

    elif isinstance(scores[0], dict):
        average = {}
        for k in scores[0]:
            average[k] = easy_reduce([s[k] for s in scores], mode=mode, skip_nan=skip_nan)

    elif isinstance(scores[0], (float, int, np.float32)):
        if skip_nan:
            scores = [x for x in scores if not np.isnan(x)]

        if mode == "mean":
            average = np.mean(scores)
        elif mode == "max":
            average = np.max(scores)
        elif mode == "median":
            average = np.median(scores)
    else:
        raise TypeError("Unsupport Data Type %s" % type(scores[0]))

    return average


def to_numpy(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, list):
        return np.array(x)
    else:
        return x


def count_parameters(model, scale=1):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / scale
