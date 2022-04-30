import torch
import gym
import matplotlib.pyplot as plt
import argparse
import os
import time
from distributedSAC.agents import *
from distributedSAC.train.runner import *
from distributedSAC.train.utils import *

'''
# add arguments in command --train/test
parser = argparse.ArgumentParser(description='Train or test neural net')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)

args = parser.parse_args()
'''

#####################  hyper parameters  ####################
ENV_ID = 'Pendulum-v0'  # environment id
RANDOM_SEED = 1  # random seed
RENDER = True  # render while training
IF_TRAIN = True
USE_GPU = False
NUM_WORKERS = 5  # NU M_WORKERS + 1 should be less or equal than num_CPUs

# RL training
TRAIN_EPISODES = 1000000  # total number of episodes for training
TEST_EPISODES = 10  # total number of episodes for testing
MAX_STEPS = 1000000  # total number of steps for each episode
WARMUP_STEPS = 800  # random action sampling in the beginning of training

GAMMA = 0.99  # discount factor
SOFT_TAU = 1e-2  # soft updating coefficient

BATCH_SIZE = 32  # update mini-batch size
UPDATE_ITR = 1  # repeated updates for single step
SOFT_Q_LR = 3e-4  # q_net learning rate
POLICY_LR = 3e-4  # policy_net learning rate
ALPHA_LR = 3e-4  # alpha learning rate
POLICY_TARGET_UPDATE_INTERVAL = 3  # delayed update for the policy network and target networks
REWARD_SCALE = 1.  # value range of reward
REPLAY_BUFFER_SIZE = 5e5  # size of the replay buffer

AUTO_ENTROPY = True  # automatically update variable alpha for entropy

###############################  demo  ####################################
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    device = "cuda:0" if torch.cuda.is_available() and USE_GPU else "cpu"

    # initialization of env
    env, state_dim, action_dim, action_range = env_ini(ENV_ID)
    print(ENV_ID)
    print('Discrete Action') if type(env.action_space) is not gym.spaces.Box else print('Continuous Action')
    print("State Dim:", state_dim, "Action Dim:", action_dim)

    # reproducible
    if RANDOM_SEED is not None:
        set_seed(env, RANDOM_SEED)

    # initialization of buffer and training info
    buffer, training_info = init_buffer_and_info(REPLAY_BUFFER_SIZE, prioritized=False)

    # initialization of agent (and experience replay buffer)
    agent = AgentSAC(
        state_dim, action_dim, action_range, buffer, training_info, BATCH_SIZE, GAMMA, ENV_ID,
        auto_entropy=AUTO_ENTROPY, target_entropy=-action_dim, soft_tau=SOFT_TAU, reward_scale=REWARD_SCALE, soft_q_lr=SOFT_Q_LR,
        policy_lr=POLICY_LR, alpha_lr=ALPHA_LR, device=device
    )
    print("Algorithm:", agent.name)

    # initialization of runner, train single actor / distributed (Ape-X style) SAC
    # runner = ApeXRunner(NUM_WORKERS, env, ENV_ID, agent, RENDER, BATCH_SIZE, TRAIN_EPISODES, TEST_EPISODES, up_itr=UPDATE_ITR,
    #                     warm_up_steps=WARMUP_STEPS, max_steps=MAX_STEPS, save_weights=True, load_weights=False, if_train=IF_TRAIN, device=device)
    runner = SingleStepRunner(env, ENV_ID, agent, RENDER, BATCH_SIZE, TRAIN_EPISODES, TEST_EPISODES, up_itr=UPDATE_ITR,
                              warm_up_steps=WARMUP_STEPS, max_steps=MAX_STEPS, save_weights=True, load_weights=False, if_train=IF_TRAIN, device=device)

    # training/testing loop
    runner.run()
