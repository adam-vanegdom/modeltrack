import json
from modeltrack.experiment import ModelTracker
from random import randint
import time


def main():
    config = {
        # "description": "Augmenting the training dataset using synonym replacement only",
        # "cuda": True,
        # "seed": 20,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "max_epochs": 10,
        "overwrite": True,
    }

    sample = ModelTracker("test-model", config=config)

    sample.start_training()
    for i in range(sample.config.max_epochs):
        sample.save_epoch_stats(
            randint(0, 100), randint(0, 100), randint(0, 100), randint(0, 100)
        )
        # time.sleep(2)
    sample.finish_training()


if __name__ == "__main__":
    main()
