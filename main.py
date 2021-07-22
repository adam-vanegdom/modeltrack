import json
from modeltrack.experiment import ModelTracker
from random import randint
import time


def main():
    config = {
        "description": "Augmenting the training dataset using synonym replacement only",
        "cuda": True,
        "gpu_device": 0,
        "seed": 20,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "max_epochs": 10,
        "overwrite": True,
    }

    sample = ModelTracker("test-model", config=config)
    # sample.print_config()

    new_config = {"seed": 10000, "batch_size": 100, "new_param": True}

    sample.store_params(new_config)

    sample.start_training()
    for i in range(sample.config.max_epochs):
        sample.save_epoch_stats(
            randint(0, 100), randint(0, 100), randint(0, 100), randint(0, 100)
        )
        time.sleep(5)
    sample.finish_training()


if __name__ == "__main__":
    main()
