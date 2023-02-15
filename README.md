
# Oveview

This folder contains the code, dataset, and results of the actual deep learning model.

# Outline

* `/VAE/utils/`: Code from the original VAE paper adding utils for training the torch model.
    * `/VAE/utils/helpers.py`: Utility functions for dealing with CUDA availability.
    * `/VAE/utils/model.py`: Abstract model, adds QoL helpers to `torch.nn.Module`.
    * `/VAE/utils/trainer.py`: Abstract trainer, adds math helpers used in the loss function.
* `/VAE/all_II_windowed.csv`: The data set, use this
* `/VAE/genGrid.py`: The grid search entry point, sets up and runs the grid search, uses James's Logging
* `/VAE/plotRun.py`: a script for turning James's run logs into plots
* `/VAE/sampleSeqs.py`: Generates a list of real sequences from the model.
* `/VAE/sequenceDataset.py`: The dataset loader. 
* `/VAE/sequenceModel.py`: Our concrete model for the AR-VAE. 
* `/VAE/sequenceTrainer_full.py`: Our concrete trainer for the AR-VAE. 
* `/VAE/train_seq_vae.py`: The one off entry point.
* `/VAE/requirements.txt`: pip requirements for running the project. (Outdated)