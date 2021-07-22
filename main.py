import json
from modeltrack.experiment import ModelTracker


def main():
    config = {
        "description": "Augmenting the training dataset using synonym replacement only",
        "cuda": True,
        "gpu_device": 0,
        "seed": 20,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "max_epoch": 100,
        "overwrite": False,
    }

    sample = ModelTracker("test-model", config=config)
    sample.print_config()

    new_config = {"seed": 10000, "batch_size": 100, "new_param": True}

    sample.set_params(new_config)
    sample.print_config()


if __name__ == "__main__":
    main()
