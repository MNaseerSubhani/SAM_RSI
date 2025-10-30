


from box import Box
from configs.base_config import base_config

config = {
    "dataset": "WHU", 
    "load_type": "soft",
    "num_points": 1,
    
    "batch_size": 1, #only support 1
    "val_batchsize": 1,
    "num_workers": 0,
    "num_epochs": 10,
    "resume": False,


    "start_lora_layer": 1,
    "lora_rank": 4,
    "match_interval": 30,
    "iou_thr": 0.1,

    "prompt": "point",
    "out_dir": "",
    "name": "base",
    "corrupt": None,
    "visual": False,
    # "model": {
    #     "type": "vit_b", 
    #     },
    "opt": {
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [8000, 15000],
        "warmup_steps": 250,
    },
}

cfg = Box(base_config)
cfg.merge_update(config)

