# Offline augmentation

This directory creates data artifacts before training. Core training does not
import these modules.

- `local_aug/prepare_captions.py`: summarize captions and produce rewrites.
- `local_aug/generate_pfe.py`: train prompt features and write public
  `features/local_pfe_sweep` arrays (one per prompt-learning config / `tprompt_idx`).
- `global_aug/generate_graphs.py`: generate GDAG JSON definitions.
- `global_aug/prepare_released_graphs.py`: compact GDAG LLM responses into the
  `{nodes, edges}` graph files.
- `image_transform/generate_features.py`: image-transformation ablation data.
- `generative/prepare_prompts.py`: create image-editing prompts.
- `generative/generate_images.py`: run Qwen image editing.
- `generative/extract_features.py`: extract features from edited images.

`local_aug/` and `global_aug/` produce the artifacts the training path consumes.
`image_transform/` and `generative/` support ablations only.
