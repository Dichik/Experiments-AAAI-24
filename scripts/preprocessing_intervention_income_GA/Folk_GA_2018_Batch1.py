# Import necessary libraries
import os
import sys

from pathlib import Path

sys.path.append(str(Path(f"{__file__}").parent.parent.parent))

from source.experiment_interface import run_exp_iter_with_disparate_impact

import copy
import warnings
import uuid
from dotenv import load_dotenv

# Ignore warnings
warnings.filterwarnings('ignore')
os.environ["PYTHONWARNINGS"] = "ignore"

from virny.utils.custom_initializers import create_config_obj
from virny.datasets import ACSIncomeDataset

from configs.constants import TEST_SET_FRACTION, EXPERIMENT_SEEDS
from configs.models_config_for_tuning import get_model_params_for_mult_repair_levels
from source.utils.db_functions import connect_to_mongodb


# Change directory if not in the correct folder
cur_folder_name = os.getcwd().split('\\')[-1]
if cur_folder_name != "fairness-variance":
   os.chdir("../..")
print('Current location: ', os.getcwd(), flush=True)

# Define Input Variables
ROOT_DIR = os.getcwd()
EXPERIMENT_NAME = 'preprocessing_intervention_income_GA'
DB_COLLECTION_NAME = 'lfr_preprocessing'
SAVE_RESULTS_DIR_PATH = os.path.join(ROOT_DIR, 'scripts', 'results', EXPERIMENT_NAME)

config_yaml_path = os.path.join(ROOT_DIR, 'scripts', 'folk_ga_2018_config.yaml')
metrics_computation_config = create_config_obj(config_yaml_path=config_yaml_path)

# # Print metrics_computation_config
print(metrics_computation_config, flush=True)

# # Define a db writer and custom fields to insert into your database
load_dotenv('./configs/secrets.env')
os.getenv("DB_NAME")

client, collection_obj, db_writer_func = connect_to_mongodb(DB_COLLECTION_NAME)

custom_table_fields_dct = {
     'session_uuid': "f998e7d0-7ccd-4e08-8f9b-93b3e771bdcb",
     # 'session_uuid': str(uuid.uuid4()),
}
print('Current session uuid: ', custom_table_fields_dct['session_uuid'], flush=True)

# # Initialize custom objects
data_loader = ACSIncomeDataset(state=['GA'], year=2018, with_nulls=False, subsample_size=15_000, subsample_seed=42)
print(data_loader.X_data.head())
print(data_loader.X_data.shape)

# Run experiment iterations
# Experiment iteration 1
exp_iter_num = 6
experiment_seed = EXPERIMENT_SEEDS[exp_iter_num - 1]
FAIR_INTERVENTION_PARAMS_LST = [
    "Ax=50, Az=10, Ay=10",
    "Ax=10, Az=50, Ay=10",
    "Ax=50, Az=50, Ay=10"
]
# default
# find right parameters (run pipeline)
custom_table_fields_dct['experiment_iteration'] = f'Exp_iter_{exp_iter_num}'
exp_iter_data_loader = copy.deepcopy(data_loader)
models_params_for_tuning = get_model_params_for_mult_repair_levels(experiment_seed)

print("Model params for tuning:\n", models_params_for_tuning, flush=True)

print("#"*20, "Experiment iteration 1", "#"*20)
run_exp_iter_with_disparate_impact(data_loader=exp_iter_data_loader,
                                   experiment_seed=experiment_seed,
                                   test_set_fraction=TEST_SET_FRACTION,
                                   db_writer_func=db_writer_func,
                                   fair_intervention_params_lst=FAIR_INTERVENTION_PARAMS_LST,
                                   models_params_for_tuning=models_params_for_tuning,
                                   metrics_computation_config=metrics_computation_config,
                                   custom_table_fields_dct=custom_table_fields_dct,
                                   with_tuning=True,
                                   save_results_dir_path=SAVE_RESULTS_DIR_PATH,
                                   verbose=True)
