# import statements
import os
import json
import time
import torch
from utils.config import process_config, update_config
from utils.logging import setup_logging


# Python API Functions
class ModelTracker:
    def __init__(self, exp_name, root_dir=None, config=None):
        self.exp_name = exp_name
        self.config = config
        self.config = process_config(exp_name, root_dir, config)
        self.logger = setup_logging(self.config.log_dir)

    def print_config(self):
        print(self.config)

    def store_params(self, new_config):
        """
        Store the hyperparameters of the model that will be used during training
        :param new_config:  new configuration to be added to experiment configuration
        :return:            nan
        """
        update_config(self.config, new_config)

        hyper_path = os.path.join(self.config.model_dir, "hyperparams.json")
        with open(hyper_path, "w") as json_file:
            json.dump(dict(self.config), json_file)

    # def store_network(self, model, loss_fn, optimizer):
    #     """

    #     :param --:
    #     :param --:
    #     :return:
    #     """
    #     self.model = model
    #     self.loss_fn = loss_fn
    #     self.optimizer = optimizer

    def start_training(self):
        """
        Signal to the modeltracker that
        :return:
        """
        # start a time to signal beginning of training
        self.logger("training has started...")
        self.start = time.time()

        # setup training storage mechanisms
        self.train_stats = {
            "train_loss": [],
            "test_loss": [],
            "train_acc": [],
            "test_acc": [],
            "dur": [],
        }
        self.best_loss = float("inf")

    def save_epoch_stats(self, train_loss, test_loss, train_acc, test_acc):
        """
        :param --:
        :param --:
        :param --:
        :param --:
        :return:
        """
        # if test_loss < self.best_loss:
        #     self.best_loss = test_loss

        self.train_stats["train_loss"].append(train_loss)
        self.train_stats["test_loss"].append(test_loss)
        self.train_stats["train_acc"].append(train_acc)
        self.train_stats["test_acc"].append(test_acc)

        epoch_time = time.time()
        self.train_stats["dur"].append(epoch_time - self.start)

        self.logger.info(
            "Epoch Complete in {}\n".format(epoch_time - self.start)
            + f"\nTraining Loss: {train_loss}, Validation Loss: {test_loss}"
            + f"\nTraining Accuracy: {train_acc}, Validation Accuracy: {test_acc}"
        )

    def save_model(self, model, epoch, optimizer, loss):
        """
        Save the state of the model in a checkpoint file
        :param model: model object
        :param epoch:
        :param optimizer:
        :param loss:
        :return:
        """
        if loss < self.best_loss:
            self.best_loss = loss
            torch.save(
                model.state_dict(),
                os.path.join(self.config.model_dir, "best_checkpoint.pt.tar"),
            )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            os.path.join(self.config.model_dir, "model_checkpoint.pt.tar"),
        )

    def finish_training(self):
        # stop timer and record the value for report

        # save all of the hyperparameters and train stats

        # produce report
        print("Model training has completed")
