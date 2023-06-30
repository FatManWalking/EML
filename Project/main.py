# Main File for the experiments
# Reads out the config yaml file and starts the experiments

import yaml
import argparse

from src.experiments import Experiment
from src.utils import init

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="config/default_config.yaml",
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
    model_name = config["config"]["model"]["model_class"]

    # Get the model class
    module = __import__("src.model", fromlist=[model_name])
    model_class = getattr(module, model_name)

    # Get the model
    model = model_class(**config["config"]["model"][model_name])

    return model


def get_data():
    """
    Get the dataloader specified in the config file.
    """

    # Get the dataloader name
    dataloader_name = config["config"]["data"]["dataset"]

    # Get the dataloader class
    module = __import__("src.dataloader", fromlist=[dataloader_name])
    dataloader_class = getattr(module, dataloader_name)

    # Get the dataloader
    dataloader = dataloader_class(**config["config"]["data"]["args"])

    return dataloader


model, dataloader = get_model(), get_data()

# Start the experiment
experiment = Experiment(model, dataloader, _wand)
experiment.run()

# Log artifacts
experiment.log_artifacts()

# Save the model
experiment.save_model()
