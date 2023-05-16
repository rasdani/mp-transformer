# might move this to a yaml or json
CONFIG = {
    "pose_dim": 3,
    "num_attention_heads": 4,
    "num_transformer_layers": 4,
    # "latent_dim": 256,
    # "latent_dim": 16,
    "latent_dim": 64,
    # "num_primitives": 8,
    "num_primitives": 4,
    # "hidden_dim": 512,
    # "hidden_dim": 64,
    "hidden_dim": 256,
    # "learn_segmentation": False,
    "learn_segmentation": True,
    "masking_slope": 1,
    "kl_weight": 1e-4,
    "mask_weight": 1e-2,
    # "kl_weight": 1e-0,
    # "kl_weight": 0,
    # "mask_weight": 1e-0,
    # "pose_weight": 1e-4,
    "pose_weight": 1,
    "segmentation_weight": 1,
    # "segmentation_weight": 1e-2,
    "lr": 1e-4,
    "batch_size": 16,
    # "batch_size": 32,
    "N": 5000,
    # "N": 20000,
    # "sequence_length": 64,
    "sequence_length": 128,
    # "epochs": 3000,
    # "epochs": 6000,
    "epochs": 1,
}

# for hyperparameter tuning with wandb sweep
SWEEP_CONFIG = {
    "method": "random",
    "metric": {
        "name": "val_loss",
        "goal": "minimize"
    },
    "parameters": {
        "pose_dim": {"values": [3]},
        "num_attention_heads": {"values": [4]},
        "num_transformer_layers": {"values": [4]},
        "latent_dim": {"values": [16, 32, 64]},
        "num_primitives": {"values": [4, 8, 16, 32]},
        "hidden_dim": {"values": [256]},
        "learn_segmentation": {"values": [True]},
        "masking_slope": {"values": [1]},
        "kl_weight": {"min": 0, "max": 1},
        "mask_weight": {"min": 0, "max": 1},
        "pose_weight": {"min": 0, "max": 1},
        "segmentation_weight": {"min": 0, "max": 1},
        "lr": {"min": 1e-4, "max": 1e-2},
        "batch_size": {"values": [16, 32, 64, 128]},
        "sequence_length": {"values": [128]},
        "N": {"values": [20000]},
        "epochs": {"value": 2}
    }
}