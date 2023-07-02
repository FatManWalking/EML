# Experiment Class is defined in experiments.py
#
# Purpose: Setting up the experiments in a modular way
#          and running them in a loop, while tracking the results
#          in a wandb project (https://wandb.ai/)
#          The experiments are defined in a yaml file

import pathlib
import yaml
import argparse
import torch
import torch.nn as nn
from .model import WeightClamper
import tqdm


class Experiment:
    def __init__(self, model, dataloader, wandb):

        self.model = model
        self.dataloader = dataloader
        self.wandb = wandb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_loader = dataloader.train_loader
        self.val_loader = dataloader.val_loader
        self.test_loader = dataloader.test_loader

    def init_loss(self):

        self.criterion = nn.CrossEntropyLoss(weight=None)

    def init_optimizer(self, general, optimizer):
        # Set the optimizer
        optim_name = optimizer["optimizer_type"]
        if optim_name == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=float(optimizer["lr"])
            )
        else:
            raise ValueError(f"Optimizer with name {optim_name} is not supported.")

        # Set the scheduler
        sched_name = optimizer["lr_scheduler"]
        if sched_name == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=general["epochs"]
            )
        else:
            raise ValueError(f"Scheduler with name {sched_name} is not supported.")

        # Figure out the clamping and put in none if it wasn't specified
        try:
            weight_clamping_params = [
                general["weight_clamping"]["min"],
                general["weight_clamping"]["max"],
            ]
        except KeyError:
            weight_clamping_params = [None, None]
        self._weightClamper = WeightClamper(*weight_clamping_params)

    def run(self):
        """
        We use wandb.config to access the config of the run.
        """
        self.model.to(self.device, dtype=torch.float32)
        self.init_optimizer(
            self.wandb.config["general"], self.wandb.config["optimizer"]
        )
        self.init_loss()

        # Run the training loop with tqdm progress bar
        pbar = tqdm.tqdm(range(self.wandb.config["general"]["epochs"]))
        for epoch in range(self.wandb.config["general"]["epochs"]):
            self.training_step()
            self.validation_step()

            # Update the scheduler
            self.scheduler.step()
            pbar.update(1)

        # Run the validation loop on the test dataset
        self.validation_step(run_on_test_dataset_instead=True)

        # Save the model
        self.save_model()

    def log_artifacts(self):

        self.wandb.log_artifact(self.model)

    def save_model(self):

        self.model.to_onnx()
        self.wandb.save("/models/*.onnx")

    def training_step(self):
        # Training step
        self.model.train()
        summed_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            # Apply weight clamping after optimization step
            self.model.apply(self._weightClamper)

            summed_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Logging
        accuracy = 100.0 * correct / total
        # Log to wandb
        self.wandb.log({"training.loss": summed_loss, "training.accuracy": accuracy})

        return summed_loss, accuracy

    def validation_step(self, run_on_test_dataset_instead=False):
        # Validation step
        self.model.eval()
        summed_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            curr_ds = self.val_loader
            if run_on_test_dataset_instead:
                curr_ds = self.test_loader
            for batch_idx, (inputs, targets) in enumerate(curr_ds):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                summed_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        # Logging
        accuracy = 100.0 * correct / total
        if run_on_test_dataset_instead:
            # Log to wandb
            self.wandb.log({"test.loss": summed_loss, "test.accuracy": accuracy})
        else:
            self.wandb.log(
                {"validation.loss": summed_loss, "validation.accuracy": accuracy}
            )

        return summed_loss, accuracy
