import wandb
import yaml

from experiment_launcher import run_experiment, single_experiment_yaml
from main_pend import *


def experiment(
    #######################################
    config_file_path: str = './config/pend.yaml',

    some_default_param: str = 'b',

    debug: bool = True,

    #######################################
    # MANDATORY
    seed: int = 41,
    results_dir: str = 'logs_pend',

    #######################################
    # OPTIONAL
    # accept unknown arguments
    **kwargs
):
    wandb.login()

    #######################################
    # MANDATORY

    results_dir: str = 'res_pend'

    with open(config_file_path, 'r') as f:
        configs = yaml.load(f, yaml.Loader)

    print('Config file content:')
    print(configs)
    wandb.init(config=configs,project="pendelum_test")
    new_full(configs)
    wandb.finish()
    print("DONE")
