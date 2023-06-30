# Experiment Class is defined in experiments.py
#
# Purpose: Setting up the experiments in a modular way
#          and running them in a loop, while tracking the results
#          in a wandb project (https://wandb.ai/)
#          The experiments are defined in a yaml file

import pathlib
import yaml
import argparse

class Experiment():

    def __init__(self):
        pass