import sys
import os
import json
import time
import torch
import glob
import argparse

from utils.config import process_config, update_config
from utils.logging import setup_logging, remove_logging
from modeltrack.report import plot_loss, produce_summary_pdf
from modeltrack import __version__

# Python API Functions
class ModelTracker:
    def __init__(self, exp_name, root_dir=None, config=None, model=None):
        self.config, self.exp_name = process_config(exp_name, root_dir, config)
        self.logger = setup_logging(self.config.log_dir)
        self.model = model

        self.logger.info(
            " *************************************** "
            + "\nThe experiment name is {}".format(self.exp_name)
            + "\nResults are saved at: {}".format(self.config.model_dir)
            + "\n*************************************** "
        )

    def print_config(self):
        print(self.config)

    def store_params(self, new_config):
        """
        Store the hyperparameters of the model that will be used during training
        :param new_config:  new configuration to be added to experiment configuration
        """
        update_config(self.config, new_config)

    def store_network(self, model):
        """
        Update the tracker with the model architecture to be used
        :param model: nn.Module type object
        """
        self.model = model

    def start_training(self):
        """
        Signal to the modeltracker that a new training run has begun
        """
        # start a time to signal beginning of training
        self.logger.info("Training has started...")
        self.start = time.time()

        self.config.current_epoch = 0

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
        Store the epoch statistics to be displayed and analyzed in output,
        and log these statictics for user
        :param train_loss:  training loss of single epoch
        :param test_loss:   training accuracy of single
        :param train_acc:   testing/validation loss of single epoch
        :param test_acc:    testing/validation accuracy of single epoch
        :return:
        """
        self.train_stats["train_loss"].append(train_loss)
        self.train_stats["test_loss"].append(test_loss)
        self.train_stats["train_acc"].append(train_acc)
        self.train_stats["test_acc"].append(test_acc)

        self.config.current_epoch += 1
        epoch_time = time.time()
        self.train_stats["dur"].append(epoch_time - self.start)

        self.logger.info(
            f"\n--------- Epoch No: {self.config.current_epoch} ---------"
            + "\nTotal Time Elapsed: {:.3f}s".format(epoch_time - self.start)
            + f"\nTraining Loss: {train_loss}, Validation Loss: {test_loss}"
            + f"\nTraining Accuracy: {train_acc}, Validation Accuracy: {test_acc}"
        )

    def save_model(self, model, epoch, optimizer, loss):
        """
        Save the state of the model in a checkpoint file
        :param model:       nn.Module object
        :param epoch:       current epoch count
        :param optimizer:   torch optimizer
        :param loss:        current validation loss
        """
        if loss < self.best_loss:
            self.best_loss = loss
            torch.save(
                model.state_dict(),
                os.path.join(self.config.check_dir, "best_checkpoint.pt.tar"),
            )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            os.path.join(self.config.check_dir, "model_checkpoint.pt.tar"),
        )

    def finish_training(self):
        """
        Save the training parameters for review and produce training report
        """
        self.logger.info("Training has ended...")

        # stop timer and record the value for report
        end = time.time()
        self.train_stats["total_dur"] = end - self.start
        self.train_stats["avg_dur"] = (
            self.train_stats["total_dur"] / self.config.current_epoch
        )

        # save the current hyperparameters used for testing
        hyper_path = os.path.join(self.config.model_dir, "hyperparams.json")
        with open(hyper_path, "w") as json_file:
            json.dump(dict(self.config), json_file)

        # generate loss curves
        plot_loss(
            self.config.model_dir,
            self.config.current_epoch,
            self.train_stats["train_loss"],
            self.train_stats["test_loss"],
        )

        # generate training summary report
        produce_summary_pdf(
            self.exp_name,
            os.path.join(self.config.model_dir, "training_loss_curve.png"),
            self.config,
            self.model,
            self.train_stats,
        )

        for handler in self.logger.handlers:
            self.logger.removeHandler(handler)


# ---- CLI ----
# The functions defined in this section are wrappers around the main Python
# API allowing them to be called directly from the terminal as a CLI
# executable/script.


def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Just a Experiment Viewer")
    parser.add_argument(
        "--version",
        action="version",
        version="modeltrack {ver}".format(ver=__version__),
    )
    parser.add_argument(
        dest="exp",
        help="experiement name to be printed",
        type=str,
        metavar="STRING",
        nargs="?",
    )
    parser.add_argument("-l", "--list", action="store_true")
    return parser.parse_args(args)


def list_experiments():
    root = os.path.join(os.getenv("HOME"), ".modeltrack")
    print("-------------------------")
    print(f"The following experiments are available at {root}")
    experiments = glob.glob(f"{root}/*")
    for exp in sorted(experiments):
        print("*** {}".format(exp.split("/")[-1]))
    print("-------------------------")


def show_logs(exp_name):
    print("-------------------------")
    print(f"Showing logs for {exp_name}")
    root = os.path.join(os.getenv("HOME"), ".modeltrack", exp_name)
    fname = os.path.join(root, "logs", "training_log.log")
    with open(fname, "r") as fin:
        print(fin.read())


def main(args):
    args = parse_args(args)
    if args.list:
        list_experiments()
    else:
        show_logs(args.exp)


def run():
    """
    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
