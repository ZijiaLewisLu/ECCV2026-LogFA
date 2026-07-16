"""Default configuration for the frame-action action-segmentation model."""

from yacs.config import CfgNode as CN


_C = CN()

# Runtime / logging
_C.aux = CN()
_C.aux.gpu = 0
_C.aux.device = "cuda"
_C.aux.mark = ""
_C.aux.runid = 0
_C.aux.debug = False
_C.aux.wandb_project = "LogFA"
_C.aux.wandb_entity = None
_C.aux.wandb_offline = False
_C.aux.wandb_watch = False
_C.aux.resume = "max"
_C.aux.skip_finished = True
_C.aux.eval_every = 200
_C.aux.print_every = 200
_C.aux.cfg_file = []
_C.aux.set_cfgs = None
_C.aux.exp = ""
_C.aux.logdir = ""
_C.aux.wandb_id = None

# EgoPER dataset
_C.dataset = "EgoPER"
_C.dname = "local_aug"
_C.recipe = None
_C.split = "a1_3v"
_C.test_split = None
_C.sr = 10
_C.test_sr = 10
_C.eval_bg = False
_C.error_vid = False
_C.data_root = None
_C.gdag_root = None
_C.local_aug_prob = 0.3
_C.global_aug_prob = 0.3
# Unified augmentation-trigger probability. -1 (default) keeps the separate
# local_aug_prob / global_aug_prob above. When >= 0, it overrides BOTH with a
# single trigger probability.
_C.taug_ratio = -1.0
# Local-PFE prompt-config index. Selects which prompt-learning config to read
# from features/local_pfe_sweep/{recipe}/{video}_{tprompt_idx}.npy (18 configs,
# 0..17). Default 0; logfa configs set this per recipe.
_C.tprompt_idx = 0
_C.local_pfe_sweep_root = None
_C.batch_size = 1

# Training
_C.optimizer = "Adam"
_C.epoch = None
_C.max_iter = 2000
_C.lr = 1e-4
_C.lr_decay = -1
_C.momentum = 0.0
_C.weight_decay = 0.0
_C.clip_grad_norm = 10.0

# Frame-action cross-attention block. block="iuu" is the default 3-stage layout.
_C.FACT = FACT = CN()
FACT.ntoken = 50
FACT.block = "iuu"
FACT.trans = False
FACT.fpos = False
FACT.cmr = 0.0
FACT.mwt = 0.0

# Input block configuration.
_C.Bi = Bi = CN()
Bi.hid_dim = 512
Bi.dropout = 0.1
Bi.a = "sca"
Bi.a_nhead = 8
Bi.a_ffdim = 512
Bi.a_layers = 6
Bi.a_dim = 128
Bi.f = "m"
Bi.f_layers = 10
Bi.f_ln = False
Bi.f_dim = 128
Bi.f_ngp = 1

# Update block configuration.
_C.Bu = Bu = CN()
Bu.hid_dim = None
Bu.dropout = None
Bu.a = "sa"
Bu.a_nhead = 8
Bu.a_ffdim = None
Bu.a_layers = 1
Bu.a_dim = None
Bu.f = "m"
Bu.f_layers = 10
Bu.f_ln = None
Bu.f_dim = None
Bu.f_ngp = None

# Temporal down/up block (unused by the "iuu" layout).
_C.BU = BU = CN()
BU.hid_dim = None
BU.dropout = None
BU.a = "sa"
BU.a_nhead = 8
BU.a_ffdim = None
BU.a_layers = 1
BU.a_dim = None
BU.f = "m"
BU.f_layers = 10
BU.f_ln = None
BU.f_dim = None
BU.f_ngp = None
BU.s_layers = 1

# Loss configuration.
_C.Loss = Loss = CN()
Loss.pc = 0.2
Loss.a2fc = 1.0
Loss.match = "o2o"
Loss.bgw = 1.0
Loss.nullw = -1.0
Loss.sw = 5.0

# Temporal masking.
_C.TM = TM = CN()
TM.use = False
TM.t = 60
TM.p = 0.1
TM.m = 5
TM.inplace = True


def get_cfg_defaults():
    return _C.clone()


RENAME_KEYS = {}
