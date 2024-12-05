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

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, logger, monitor, video_recorder, seed, env_cfg, step, max_episode_steps, nb_eval_episodes, imaginary=True,  record=False, nb_recorded_episodes=20):
    eval_env = initiate_class(env_cfg['name'], env_cfg['settings'])
    eval_env.reset(seed+step)
    average_episode_return, list_episode_reward = 0, []
    video_recorder.init(enabled=record)

    for episode in tqdm(range(0, nb_eval_episodes), desc =f"{'Imaginary' if imaginary else 'Real'} Model Evaluation"):
        obs = eval_env.reset(imaginary=imaginary)
        done = False
        episode_reward, episode_step = 0, 0
        if episode < nb_recorded_episodes:
            video_recorder.record(eval_env, render_cfg={'imaginary': imaginary}, episode=episode, returns=episode_reward, step=episode_step)

        while not done and (episode_step < max_episode_steps):
            action = policy.act(obs)
            obs, reward, done, _, info = eval_env.step(action, imaginary=imaginary)
            episode_reward += reward
            episode_step += 1
            if episode < nb_recorded_episodes:
                video_recorder.record(eval_env, render_cfg={'imaginary': imaginary}, episode=episode, returns=episode_reward, step=episode_step)

        average_episode_return += episode_reward
        list_episode_reward.append(episode_reward)

        monitor.collect_episode_performance(info)

    average_episode_return /= nb_eval_episodes
    logger.log(f"eval/{'imaginary' if imaginary else 'real'}_episodic_return_avg", average_episode_return, step)
    logger.log(f"eval/{'imaginary' if imaginary else 'real'}_episode_return_std", np.array(list_episode_reward).std(), step)
    logger.log_histogram(f"eval/{'imaginary' if imaginary else 'real'}_episode_reward_distribution", np.array(list_episode_reward), step)
    if not imaginary:
        logger.dump(step)
    video_recorder.save(f"{step}_{'imaginary' if imaginary else 'real'}.mp4")
    monitor.dump_end_eval(logger, step)


def run(args):
    with open(args.cfg_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    today_date = time.strftime("%Y%m%d")
    today_time = time.strftime("%H%M%S")
    start_time = time.time()
    exp_folder = f"./experiments/{today_date}/{today_time}_{cfg['experiment']}/"
    os.makedirs(exp_folder+"models/", exist_ok=True)
    os.makedirs(exp_folder+"videos/", exist_ok=True)
    os.makedirs(exp_folder+"checkpoint/", exist_ok=True)

    # Save experiment configuration
    with open(f'{exp_folder}config.yaml', 'w') as file:
        documents = yaml.dump(cfg, file)
    
    # Logger
    logger = Logger(exp_folder,
                    save_tb=cfg['train']['save_tensorboard'],
                    log_frequency=cfg['train']['log_frequency'],
                    agent=cfg['agent'])
    imaginary_video_recorder = VideoRecorder(exp_folder + "videos/") #self.work_dir if cfg.save_video else None)
    real_video_recorder = VideoRecorder(exp_folder + "videos/") #self.work_dir if cfg.save_video else None)
    imaginary_eval_monitor = EvalMonitor(cfg['logger']['imaginary'])
    real_eval_monitor = EvalMonitor(cfg['logger']['real'])
    
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
    policy_cfg = cfg['policy']['settings']
    policy_cfg['obs_dim'] = env.observation_space
    policy_cfg['action_dim'] = env.action_space.n
    logging.info('Experiment configuration')
    logging.info('Observation space:',env.observation_space)
    logging.info('Action space:', policy_cfg['action_dim'])

	# Initialize policies
    policy = initiate_class(cfg['policy']['name'], policy_cfg, multiple_arguments=False)
    if 'load' in policy_cfg.keys():
        policy.load(policy_cfg['load'])

    # Buffer initialization and data collection
    memory_cfg = cfg['memory']
    memory_cfg['settings']['obs_dim'] = env.observation_space
    memory_cfg['settings']['action_dim'] = env.action_space
    memory = initiate_class(memory_cfg['name'], memory_cfg['settings'], multiple_arguments=False)
    priority_weight_increase = (1 - memory_cfg['settings']['priority_weight']) / (cfg['train']['train_steps'])
    
    data_collection_noise(env=env,
                          policy=policy,
                          memory=memory, 
                          data_collection_steps=cfg['train']['data_collection_steps'],
                          reset_noise_steps=cfg['train']['replay_frequency'],
                          reward_clip=cfg['train']['reward_clip'],
                          epsilon=cfg['train']['exploration_epsilon'])

    step, episode_step, episode, episode_reward, done = 0, 1, 0, 0, True
    start_time = time.time()
    for step in range(cfg['train']['train_steps'] + 1):
        # Evaluate agent
        if step % cfg['train']['eval_frequency'] == 0:
            logger.log('eval/episode', episode, step)
            # Eval imaginary
            eval_policy(policy, 
                        logger, 
                        imaginary_eval_monitor, 
                        imaginary_video_recorder, 
                        ENV_SEED, 
                        env_cfg, 
                        step, 
                        cfg['train']['max_episode_steps'], 
                        cfg['train']['nb_eval_episodes'],
                        imaginary=True,
                        record=cfg['train']['record_imaginary']
                        )
            # Eval real
            eval_policy(policy, 
                        logger, 
                        real_eval_monitor, 
                        real_video_recorder, 
                        ENV_SEED, 
                        env_cfg, 
                        step, 
                        cfg['train']['max_episode_steps'], 
                        cfg['train']['nb_eval_episodes'],
                        imaginary=False,
                        record=cfg['train']['record_real']
                        )
            
        # log end of episode or truncation and reset env
        if done or (episode_step >= cfg['train']['max_episode_steps']):
            if step > 0:
                logger.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                logger.dump(step)

            logger.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

            logger.log('train/episode', episode, step)

        # sample action for data collection
        action = policy.act(obs)

        next_obs, reward, done, _, _= env.step(action, imaginary=True)

        episode_reward += reward
        if cfg['train']['reward_clip'] > 0:
            reward = max(min(reward, cfg['train']['reward_clip']), -cfg['train']['reward_clip'])  # Clip rewards
        memory.append(obs, action, reward, done)

        obs = next_obs
        episode_step += 1

        # Train
        memory.priority_weight = min(memory.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1
        if step % cfg['train']["replay_frequency"] == 0:
            policy.reset_noise()
            policy.update(memory, logger, step)

        # Update target network
        if step % cfg['train']["target_update"] == 0:
            policy.update_target_net()

        if step % cfg['train']['checkpoint_frequency'] == 0:
            policy.save(exp_folder+"checkpoint/", f"checkpoint_{step}.ph")
        
    policy.save(exp_folder+"models/")

def data_collection_noise(env, 
                          memory, 
                          policy, 
                          data_collection_steps, 
                          reset_noise_steps,
                          reward_clip,
                          epsilon=0):
    done = True

    for step in tqdm(range(0, data_collection_steps), desc ="Replay Buffer filling"):
        if done:
            obs = env.reset()
            done = False

        if step % reset_noise_steps==0:
            policy.reset_noise()

        # sample action for data collection
        action = policy.act_e_greedy(obs, epsilon)
        
        #obs, reward, terminated, truncated, info
        next_obs, reward, done, _, _= env.step(action, imaginary=True)
        if reward_clip > 0:
            reward = max(min(reward, reward_clip), -reward_clip) 
        # append(state, action, reward, terminal):
        memory.append(obs, action, reward, done)
        obs = next_obs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", required=True)
    args = parser.parse_args()
    run(args)