import torch
import yaml
import argparse
from model.environments.env import FullPartialObsEnviroment
from random import randint, choices
import numpy as np
import pickle
from model.agents.Rand import RandPolicy
from model.agents.DQN import Rainbow
from tqdm import tqdm

def generate_obs(env, policy, n_steps, max_ep_len=100, epsilon=0):
    #env = FullPartialObsEnviroment(env)
    actions_set = np.zeros(n_steps)
    obs_set = np.zeros([n_steps, 7*7 + 1])
    done_set = np.zeros(n_steps)
    ep_indices_set = []

    terminated = False
    obs, ep_len, action, reward = env.reset(), 1, -1, 0
    #print(obs.keys())
    #print("full obs", obs['image'].shape)
    #print("partial obs", obs['partial_image'].shape)
    ep_indices_set.append([0, -1])

    # Main Loop
    i = 0
    pbar = tqdm(total=n_steps)
    while i < n_steps:
        if isinstance(policy, RandPolicy):
            policy.reset()

        # Save enviroment configurations
        obs_set[i, :49] = obs['partial_image'][:,:,0].flatten()
        obs_set[i, 49] = reward
        done_set[i] = terminated
        actions_set[i] = action

        # Take action
        #d = {'image': obs["full_observation"], 'direction': obs["direction"], 'mission': obs["mission"]}
        if epsilon > 0 and isinstance(policy, Rainbow):
            action = policy.act_e_greedy(obs, epsilon)
        else:
            action = policy.act(obs)

        obs, reward, terminated, _, _ = env.step(action)
        
        ep_len += 1

        if terminated or (ep_len >= max_ep_len):
            i += 1
            pbar.update(1)
            if i < n_steps:
                obs_set[i, :49] = obs['partial_image'][:,:,0].flatten()
                obs_set[i, 49] = reward
                done_set[i] = terminated
                actions_set[i] = action

                obs, action, reward, terminated = env.reset(), -1, 0, False
                ep_len = 1
                ep_indices_set[-1][1] = i
                if not i == (n_steps - 1):
                    ep_indices_set.append([i + 1, -1])

        i += 1
        pbar.update(1)

    pbar.close()

    dictionary = {'obs_set':obs_set, 'actions':actions_set, 'Terminated': done_set, 'ep_indexs': np.array(ep_indices_set)}
    return dictionary

def save_pickle(save_path, data):
    # save dictionary to pickle file
    with open(save_path, "wb") as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

def load_pickle(load_path):
    # load dictionary to pickle file
    with open(load_path, "rb") as file:
        loaded_dict = pickle.load(file)
    return loaded_dict


