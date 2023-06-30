import wandb
import yaml


def init(config=None):
    # Initialize wandb
    # Config contains all info including the name of the run and notes
    config = load_config(config)
    wandb.init(
        project="eml",
        # Set the name of the run
        name=config["name"],
        # Set the notes of the run
        notes=config["notes"],
        # Set the config of the run excluding the name and notes
        config=config["config"],
    )

    return wandb


def load_config(config) -> dict:
    # Load config from yaml file
    if config is None:
        with open("default_config.yaml", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        with open(config, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    return config
