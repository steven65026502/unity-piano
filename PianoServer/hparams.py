from torch import cuda, device as d
# get device
if cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = d(dev)
# default hparams for the model
hparams = {
    "d_model": 128,
    "num_layers": 3,
    "num_heads": 8,
    "d_ff": 512,
    "max_rel_dist": 1024,
    "max_abs_position": 0,
    "bias": True,
    "dropout": 0.1,
    "layernorm_eps": 1e-6
}