# Experiment Class is defined in experiments.py
#
# Purpose: Setting up the experiments in a modular way
#          and running them in a loop, while tracking the results
#          in a wandb project (https://wandb.ai/)
#          The experiments are defined in a yaml file

import pathlib
import yaml
import argparse


class Experiment:
    def __init__(self, model, dataloader, wandb):

        self.model = model
        self.dataloader = dataloader
        self.wandb = wandb

    def run(self):
        """
        We use wandb.config to access the config of the run.
        """

        # Train the model
        self.model.train(self.dataloader)

        # Evaluate the model
        self.model.evaluate(self.dataloader)

        # Log the results
        self.wandb.log(self.model.results)

        # Save the model
        self.model.save()

        # Save the results
        self.model.save_results()

        # Save the config
        self.model.save_config()

    def log_artifacts(self):

        self.wandb.log_artifact(self.model)

    def save_model(self):

        self.model.to_onnx()
        self.wandb.save("/models/*.onnx")
