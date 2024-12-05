import numpy as np
import torch
import argparse
import time
import os
import yaml
import logging
from tqdm import tqdm
from utils.logger import Logger, EvalMonitor
from utils.train_utils import initiate_class
from utils.video import VideoRecorder
from generate_data import * 
from utils.train_dkf_utils import *
from torch import optim
from pathlib import Path
import shutil 


def run(args):
    with open(args.cfg_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    today_date = time.strftime("%Y%m%d")
    today_time = time.strftime("%H%M%S")
    start_time = time.time()
    exp_folder = f"./DKF_trainings/{today_date}/{today_time}_{cfg['experiment']}/"
    os.makedirs(exp_folder+"models/", exist_ok=True)
    os.makedirs(exp_folder+"checkpoint/", exist_ok=True)

    # Save experiment configuration
    with open(f'{exp_folder}config.yaml', 'w') as file:
        documents = yaml.dump(cfg, file)
    
    # Logger
    logger = Logger(exp_folder,
                    save_tb=cfg['train']['save_tensorboard'],
                    log_frequency=cfg['train']['log_frequency'],
                    agent=cfg['agent']) 

    #logger.log_histogram(tensor, step)
    #logger.log()

    # Env init
    env_cfg = cfg['env']
    env = initiate_class(env_cfg['name'], env_cfg['settings'])

    # Set seeds
    ENV_SEED = env_cfg['seed']
    env.reset(seed=ENV_SEED)
    env.action_space.seed(ENV_SEED)
    torch.manual_seed(ENV_SEED)
    np.random.seed(ENV_SEED)

    # Observations and actions space shapes
    expert_policy_cfg = cfg['expert_policy']['settings']
    expert_policy_cfg['obs_dim'] = env.observation_space
    expert_policy_cfg['action_dim'] = env.action_space.n

	# Initialize policies

    rand_policy = initiate_class(cfg['rand_policy']['name'], None, multiple_arguments=False)
    expert_policy = initiate_class(cfg['expert_policy']['name'], expert_policy_cfg, multiple_arguments=False)

    # Load trained experted policy
    if 'load' in cfg['expert_policy'].keys():
        print("Loading expert policy weights...")
        expert_policy.online_net.load_state_dict(torch.load(cfg['expert_policy']['load']))

    os.makedirs(cfg['data_generation']['data_path'], exist_ok=True)

    expert_file_path = cfg['data_generation']['data_path']+'/'+cfg['data_generation']['expert_data_file']
    rand_file_path = cfg['data_generation']['data_path']+'/'+cfg['data_generation']['rand_data_file']
    
    expert_n_steps = cfg['data_generation']['expert_n_steps']
    rand_n_steps = cfg['data_generation']['rand_n_steps']
    max_ep_len = cfg['data_generation']['max_ep_len']
    epsilon = cfg['data_generation']['epsilon_expert']

    if not os.path.isfile(expert_file_path):
        print("Expert data doesn't exist. Generating...")
        expert_data_dict = generate_obs(env, expert_policy, expert_n_steps, max_ep_len=max_ep_len, epsilon = epsilon)
        expert_dataset = fix_dataset_format(expert_data_dict)
        save_pickle(expert_file_path, expert_dataset)

    elif cfg['data_generation']['regenerate_expert_data']:
        print("Overwriting existing expert data...")
        expert_data_dict = generate_obs(env, expert_policy, expert_n_steps, max_ep_len=max_ep_len, epsilon = epsilon)
        expert_dataset = fix_dataset_format(expert_data_dict)
        save_pickle(expert_file_path, expert_dataset)

    else:
        print("Loading already existing expert data")
        expert_dataset = load_pickle(expert_file_path)


    if not os.path.isfile(rand_file_path):
        print("Random data doesn't exist. Generating...")
        rand_data_dict = generate_obs(env, rand_policy, rand_n_steps, max_ep_len=max_ep_len, epsilon = epsilon)
        rand_dataset = fix_dataset_format(rand_data_dict)
        save_pickle(rand_file_path, rand_dataset)

    elif cfg['data_generation']['regenerate_rand_data']:
        print("Overwriting existing random data...")
        rand_data_dict = generate_obs(env, rand_policy, rand_n_steps, max_ep_len=max_ep_len, epsilon = epsilon)
        rand_dataset = fix_dataset_format(rand_data_dict)
        save_pickle(rand_file_path, rand_dataset)

    else:
        print("Loading already existing random data")
        rand_dataset = load_pickle(rand_file_path)

    # Se mezclan los datos de la política random y experta
    print("Padding datasets...")
    expert_dataset, rand_dataset = pad_datasets(expert_dataset, rand_dataset)

    print(f"Num expert episodes: {len(expert_dataset['obs'])}")
    print(f"Num random episodes: {len(rand_dataset['obs'])}")

    print("Mixing datasets...")
    mixed_dataset, obs_map = mix_datasets(expert_dataset, rand_dataset, replace_dict=True)

    # Save experiment configuration
    with open(f'config/obs_map.yml', 'w') as file:
        documents = yaml.dump(float_keys_to_str(obs_map), file)

    train_dataset, val_dataset = train_test_split_(mixed_dataset, test_size=0.2, random_state=40)

    # Se crea el objeto de dataset 
    train_dataset = MinigridData(train_dataset)
    val_dataset = MinigridData(val_dataset)

    cfg_train = cfg['train']

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg_train['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg_train['batch_size'], shuffle=True)

    cfg_train['minigrid_encoder']['settings']['action_dim'] = env.action_space.n

    cfg_train['obs_map'] = obs_map

    if cfg_train['checkpoint']['load']:
        model_state = torch.load(cfg_train['checkpoint']['path'])

        model = initiate_class(cfg_train['dkf']['name'], cfg_train, multiple_arguments=False)
        model.load_state_dict(model_state['model'])

    else:
        model = initiate_class(cfg_train['dkf']['name'], cfg_train, multiple_arguments=False)
        lr = cfg_train['lr']
        amsgrad = cfg_train['amsgrad']
        optimizer = optim.AdamW(model.parameters(), lr=lr, amsgrad=amsgrad)
        model_state = {'model': model, 'optimizer':optimizer, 'epoch':0}

    cfg_train['checkpoint_path'] = exp_folder+"checkpoint/"
    cfg_train['final_model_path'] = exp_folder+"models/"

    print("Starting training...")
    model = train(model_state,
                  train_loader,
                  cfg_train,
                  logger=logger,
                  val_loader=val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", required=True)
    args = parser.parse_args()
    run(args)