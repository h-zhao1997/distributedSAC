import random
import numpy as np
import torch
import gym
from multiprocessing import managers
from distributedSAC.train.buffer import *
import matplotlib.pyplot as plt
import os
import time
import datetime


class RunningManager(managers.BaseManager):
    pass

def set_seed(env, random_seed):
    env.seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


def env_ini(env_id):
    env = gym.make(env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = env.action_space.high[0]  # scale action, [-action_range, action_range]]
    return env, state_dim, action_dim, action_range


class Traininginfo:
    def __init__(self):
        self.worker_done = 0
        self.time = []
        self.reward = []
        self.start_time = None

    def end_train(self):
        self.worker_done += 1

    def num_end_workers(self):
        return self.worker_done

    def add_episode_info(self, time, reward):
        self.time.append(time)
        self.reward.append(reward)

    def save_training_info(self, date_time, alg_name, env_id):
        path = os.path.join('model_and_image', '_'.join([alg_name, env_id])) + '/' + date_time + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        

    def plot_time_reward(self, date_time, alg_name, env_id):
        if len(self.reward) >= 2:
            plt.plot(self.time[-2:], self.reward[-2:], 'orange')
            plt.xlabel('time')
            plt.ylabel('reward')
            plt.title(alg_name + '-' + env_id)
            plt.draw()
            path = os.path.join('model_and_image', '_'.join([alg_name, env_id])) + '/' + date_time + '/'
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(os.path.join(path, '_'.join([alg_name, env_id])))

    def plot_episode_reward(self, date_time, alg_name, env_id):
        if len(self.reward) >= 2:
            plt.plot([len(self.reward) - 1, len(self.reward)], self.reward[-2:], 'orange')
            plt.xlabel('episode')
            plt.ylabel('reward')
            plt.title(alg_name + '-' + env_id)
            plt.draw()
            path = os.path.join('model_and_image', '_'.join([alg_name, env_id])) + '/' + date_time + '/'
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(os.path.join(path, '_'.join([alg_name, env_id])))

    def add_start_time(self, t):
        if self.start_time is None:
            self.start_time = t

    def get_start_time(self):
        return self.start_time


def init_buffer_and_info(buffer_size, prioritized=False):
    # initialization of multiprocessing manager
    manager = RunningManager()
    if prioritized:
        manager.register('Buffer', PrioritizedReplayBuffer)
    else:
        manager.register('Buffer', ReplayBuffer)
    manager.register('Traininginfo', Traininginfo)
    manager.start()
    buffer = manager.Buffer(buffer_size)
    training_info = manager.Traininginfo()
    return buffer, training_info