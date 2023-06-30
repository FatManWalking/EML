# Main File for the experiments
# Reads out the config yaml file and starts the experiments

import yaml
import argparse

from src.experiments import Experiment
from utils import init

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="default_config.yaml",
    help="Path to the config file.",
)
parser.add_argument(
    "--wandb",
    type=str,
    default="wandb",
    help="Path to the wandb directory.",
)

args = parser.parse_args()

# Initialize wandb
_wand = init(config=args.config)

# Read config file
with open(args.config, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def get_model():
    """
    Get the model specified in the config file.
    """

    # Get the model name
    model_name = config["model"]["name"]

    # Get the model class
    model_class = getattr(__import__("src.model"), model_name)

    # Get the model
    model = model_class(**config["model"]["args"])

    return model


def get_data():
    """
    Get the dataloader specified in the config file.
    """

    # Get the dataloader name
    dataloader_name = config["dataloader"]["name"]

    # Get the dataloader class
    dataloader_class = getattr(__import__("src.dataloader"), dataloader_name)

    # Get the dataloader
    dataloader = dataloader_class(**config["dataloader"]["args"])

    return dataloader


model, dataloader = get_model(), get_data()

# Start the experiment
experiment = Experiment(model, dataloader, _wand)
experiment.run()

# Log artifacts
experiment.log_artifacts()

# Save the model
experiment.save_model()
