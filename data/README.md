# Release data

The data packages are distributed on
[Google Drive](https://drive.google.com/drive/folders/1TunJuwp93bkb4EkTU2VMeA1EDGzsvRzF)
and, once extracted, live under `data/` (they are ignored by Git). The loader
auto-detects a package beside or above the code; an extracted copy elsewhere is
selected with `--data-root` or `LOGFA_DATA_ROOT`.

## Path configuration

Data-root priority:

1. `--data-root /path/to/LogFA-EgoPER-v1`
2. `LOGFA_DATA_ROOT`
3. auto-detected `data/LogFA-EgoPER-v1` beside or directly above the code

Every path used by a loader is derived from the configured data root; no
machine-specific path is required. Training outputs default to the repository
root — set `LOGFA_OUTPUT_ROOT` to place logs elsewhere.

## LogFA-EgoPER-v1

```text
LogFA-EgoPER-v1/
├── annotations/
│   ├── mapping.txt
│   ├── groundTruth/{video}.txt
│   └── splits/{recipe}/{split}.{train,test}
├── captions/{video}_rewritten.jsonl
├── features/
│   ├── siglip2/{video}.npy
│   └── local_pfe_sweep/{recipe}/{video}_{tprompt_idx}.npy
├── gdag/{recipe}.json
├── weights/fact/logfa/{recipe}/seed{0..3}/{model.pth,config.yaml,metadata.json}
└── manifest.json
```

Feature formats:

- `siglip2/{video}.npy`: float array `[T_full, 1024]` (base SigLIP features).
- `local_pfe_sweep/{recipe}/{video}_{tprompt_idx}.npy`: float array
  `[T_train, K, 1024]`, aligned to the 10-frame training sample rate. 18
  prompt-learning configs per video (`tprompt_idx` 0..17); `K` varies by config.
  Select one via `cfg.tprompt_idx`.

Local Aug and LogFA consume the local PFE sweep. Global Aug consumes only base
features and the GDAG definitions in `gdag/{recipe}.json` (resolved from the data
root; override with `--gdag-root`). The sweep can be rebuilt from
`captions/{video}_rewritten.jsonl` with `augmentation/local_aug/generate_pfe.py`
— see the README.

Weights: 20 LogFA checkpoints (5 recipes x 4 seeds), each a `model.pth` with its
`config.yaml` and `metadata.json`.

## LogFA-EgoProceL-v1

EgoProceL data package (recipes `ETENT`, `PC_assemble`, `PC_disassemble`,
`MECANNO`, split `split1`). Same layout, and it ships `annotations/`,
`captions/`, and `features/siglip2/`. Generate the GDAG task graphs, local PFE
sweep, and weights with the steps in the README. See
[LogFA-EgoProceL-v1/README.md](LogFA-EgoProceL-v1/README.md).
