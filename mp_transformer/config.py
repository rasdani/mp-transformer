# might move this to a yaml or json
CONFIG = {
    "pose_dim": 6,
    "num_attention_heads": 4,
    "num_transformer_layers": 4,
    # "latent_dim": 8,
    # "latent_dim": 16,
    #  "latent_dim": 32,
    # "latent_dim": 48,
    # "latent_dim": 64,
    #  "latent_dim": 128,
     "latent_dim": 256,
    #"num_primitives": 4,
    "num_primitives": 6,
    # "num_primitives": 8,
    # "hidden_dim": 16,
    #  "hidden_dim": 32,
    # "hidden_dim": 48,
    # "hidden_dim": 64,
    #  "hidden_dim": 128,
     "hidden_dim": 256,
    "learn_segmentation": True,
    "masking_slope": 1,
    # "masking_slope": 0.75,
    # "masking_slope": 0.5,
    # "kl_weight": 1e-5,
    # "kl_weight": 5e-3,
    # "kl_weight": 1e-2,
    # "kl_weight": 2e-2,
    "kl_weight": 1e-3,
    # "kl_weight": 1e-4,
    "anneal_start": 10,
    # "anneal_start": 5,
    # "anneal_start": 0,
    "anneal_end": 50,
    # "anneal_end": 30,
    # "anneal_end": 20,
    # "anneal_end": 15,
    # "anneal_end": 0,
    "cycle_length": None,
    # "cycle_length": 100,
    # "cycle_length": 200,
    # "durations_weight": 1e-6,
    #"durations_weight": 1e-4,
    "durations_weight": 0,
    # "durations_weight": 1e-5,
    "lr": 1e-4,
    # "lr": 5e-4,
    "batch_size": 8,
    # "batch_size": 16,
    "N_train": 200000,
    # "N_train": 2,
    "N_val": 40000,
    # "N_val": 2,
    "sequence_length": 128,
    #"epochs": 200,
    # "epochs": 250,
    # "epochs": 230,
    # "epochs": 3000,
    # "epochs": 1000,
    # "epochs": 300,
    "epochs": 500,
    # "epochs": 800,
    # "epochs": 400,
    # "epochs": 5,
    # "epochs": 1,
    # "run_name": "fresh-Transformer",
    # "run_name": "smol-Transformer",
    # "run_name": "resume-Transformer",
    #"run_name": "smol-Transformer",
    #"run_name": "notrafo-Transformer",
    # "run_name": "notrafo-sigmoid-Transformer",
    # "run_name": "noanneal-sigmoid-Transformer",
    # "run_name": "sigmoid-Transformer",
    # "run_name": "midKL-Transformer",
    "run_name": "lowKL-Transformer",
    # "run_name": "cyclical-Transformer",
    # "run_name": "nosigmoid-Transformer",
    # "run_name": "nosigmoid-Transformer",
    # "run_name": "relu-sigmoid-Transformer",
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
