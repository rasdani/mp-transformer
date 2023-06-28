# might move this to a yaml or json
CONFIG = {
    "pose_dim": 3,
    "num_attention_heads": 4,
    "num_transformer_layers": 4,
    # "latent_dim": 16,
    "latent_dim": 32,
    # "latent_dim": 64,
    # "latent_dim": 128,
    "num_primitives": 6,
    # "num_primitives": 8,
    # "hidden_dim": 16,
    "hidden_dim": 32,
    # "hidden_dim": 128,
    # "hidden_dim": 256,
    # "hidden_dim": 512,
    "learn_segmentation": True,
    "masking_slope": 1,
    # "masking_slope": 0.75,
    # "masking_slope": 0.5,
    # "kl_weight": 1e-5,
    "kl_weight": 1e-2,
    # "kl_weight": 1e-3,
    # "kl_weight": 1e-4,
    "anneal_start": 10,
    "anneal_end": 50,
    # "cycle_length": 1000,
    "cycle_length": None,
    # "kl_weight": 1e-4,
    # "durations_weight": 1e-6,
    "durations_weight": 1e-4,
    # "durations_weight": 1e-5,
    "lr": 1e-4,
    # "lr": 5e-4,
    "batch_size": 8,
    # "batch_size": 16,
    "N_train": 200000,
    "N_val": 40000,
    "sequence_length": 128,
    # "epochs": 250,
    # "epochs": 3000,
    "epochs": 500,
    # "epochs": 100,
    # "epochs": 1,
    # "run_name": "fresh-Transformer",
    # "run_name": "smol-Transformer",
    "run_name": "resume-Transformer",
}

# for hyperparameter tuning with wandb sweep
SWEEP_CONFIG = {
    "method": "random",
    "metric": {"name": "best_val_loss", "goal": "minimize"},
    "parameters": {
        "pose_dim": {"value": 3},
        "num_attention_heads": {"value": 4},
        "num_transformer_layers": {"value": 4},
        "latent_dim": {"values": [32, 64, 128]},
        "num_primitives": {"values": [4, 6, 8]},
        "hidden_dim": {"values": [128, 256, 512]},
        "learn_segmentation": {"value": True},
        # "masking_slope": {"value": 1.0},
        "masking_slope": {"values": [0.4, 0.5, 0.75, 1.0]},
        "kl_weight": {"values": [1e-5, 1e-4, 1e-3, 1.0]},
        "anneal_start": {"value": 10},
        "anneal_end": {"value": 50},
        # "durations_weight": {"values": [1e-4, 1e-3, 1e-2, 1e-1]},
        "durations_weight": {"values": [1e-6, 1e-5, 1e-4, 5e-4]},
        # "lr": {"values": [1e-5, 1e-4, 1e-3, 1e-2]},
        "lr": {"values": [1e-4, 5e-4]},
        "batch_size": {"values": [8, 16]},
        "sequence_length": {"value": 128},
        "N_train": {"value": 200000},
        "N_val": {"value": 40000},
        "epochs": {"value": 200},
    },
    "early_terminate": {"type": "hyperband", "min_iter": 50, "max_iter": 200},
}
