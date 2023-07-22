"""Hooked this up to pre-commit, so it runs on every commit."""
import unittest

from mp_transformer.config import CONFIG
from mp_transformer.train import main


class TestTrain(unittest.TestCase):
    """Test train.py"""

    def test_main(self):
        """Run minimal training."""
        minimal_config = {
            "latent_dim": 4,
            "num_primitives": 2,
            "hidden_dim": 4,
            "batch_size": 3,
            "sequence_length": 5,
            "N_train": 4,
            "N_val": 4,
            "epochs": 2,
        }
        CONFIG.update(minimal_config)

        # TODO: disable GPU on HPC cluster login node
        main(CONFIG, debug=True)
