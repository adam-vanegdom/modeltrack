==========
modeltrack
==========
Tool to track all PyTorch model config and architecture and prepare training summary reports

***************
Usage Guide
***************
The following will explain how to use the install the tool to your environment, and use the tool within a Jupyter Notebook 

Installation
"""""""""""""""""
To install the tool simply run the following command in your terminal:

::

   pip install --upgrade modeltest

Quick Start
"""""""""""""""""
To begin using the tool you must import the package into your notebook by adding the following: 

::

   import modeltrack.experiment as exp

After importing the experiment module you will be able to create a new tracker for your model:

::

   tracker = exp.ModelTracker('model-name', config={"max_epochs":100})

When instantiating the new ModelTracker, you must pass in the name of the experiment being run as well as the model configuration. Optionally, you can pass a directory to specify where the tracking output should be stored. The following 
configuration variables must be set, along with any other model-specific configuration:

::

      config = {
         "batch_size": [INT],
         "learning_rate": [FLOAT],
         "max_epochs": [INT],
         "overwrite": [BOOL] - if set to True, most recent experiment with that name will be overwritten
      }

The ModelTracker object has four functions that are useful during training:

- ``tracker.start_training()``:

            | Signal to the ModelTracker that a new training session has begun. Re-initializes parameter watchers

- ``tracker.save_epoch_stats(train_loss, test_loss, train_acc, test_acc)``: 

            | Store the epoch statistics to be displayed and analyzed in output, and automatically log values


            :train_loss:  training loss of single epoch
            :test_loss:   training accuracy of single
            :train_acc:   testing/validation loss of single epoch
            :test_acc:    testing/validation accuracy of single epoch

- ``tracker.save_model(model, epoch, optimizer, loss)``: 

            | Save the state of the model in a checkpoint file

            :model:       nn.Module object
            :epoch:       current epoch count
            :optimizer:   torch optimizer
            :loss:        current validation loss

- ``tracker.finish_training(model=None)``:

            | Save the training parameters for review and produce training report
      
            :model: [*Default=None*] current nn.Module model being used at end of training 

 





