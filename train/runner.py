import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from copy import deepcopy
from distributedSAC.train.utils import *
from distributedSAC.train.buffer import *
from torch.multiprocessing import Pool

class ApeXRunner:
    def __init__(self, num_workers, env, env_id, agent, render, batch_size, train_episode, test_episode, up_itr=1,
                 warm_up_steps=0, max_steps=1000000, save_weights=False, load_weights=False, if_train=True, device='cpu'):
        self.num_workers = num_workers
        self.env = env
        self.env_id = env_id
        self.agent = agent
        self.render = render
        self.batch_size = batch_size
        self.episode = train_episode
        self.train_episode = train_episode
        self.test_episode = test_episode

        self.up_itr = up_itr
        self.warm_up_steps = warm_up_steps
        self.max_steps = max_steps
        self.save_weights = save_weights
        self.if_train = if_train
        self.load_weights = load_weights
        self.device = device

        self.date_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    def worker_process(self, render, worker_num):
        env = gym.make(self.env_id)
        local_buffer_size = int(self.batch_size / self.num_workers)
        local_buffer = ReplayBuffer(local_buffer_size)
        local_policy_net = deepcopy(self.agent.policy_net).to('cpu')
        frame_idx = 0
        print('Worker', worker_num, 'is ready!')
        self.agent.training_info.add_start_time(time.time())
        for i_episode in range(1, self.train_episode + 1, 1):
            state = env.reset().astype(np.float32)
            episode_reward = 0
            for step in range(self.max_steps):
                if render:
                    env.render()
                if frame_idx >= int (self.warm_up_steps / self.num_workers):
                    action = local_policy_net.get_action(torch.Tensor(state).unsqueeze(0).to('cpu'))
                else:
                    action = local_policy_net.sample_action()

                next_state, reward, done, _ = env.step(action)
                next_state = next_state.astype(np.float32)
                mask = 0.0 if done else self.agent.gamma  # mask = (1 - done) * gamma
                transition = (state, action, reward, next_state, mask)
                local_buffer.push(transition)
                state = next_state
                episode_reward += reward
                frame_idx += 1

                if local_buffer.len() >= local_buffer_size:
                    self.agent.buffer.push_batch(local_buffer.buffer)
                    local_buffer.clean()

                local_policy_net.load_state_dict(self.agent.policy_net.state_dict())


                if done:
                    break
            end_episode_time = time.time() - self.agent.training_info.get_start_time()
            print(
                'Worker {} Training  | Episode: {}/{}  | Episode Reward: {:.4f}  | Episode step: {:.4f}  | Running Time: {:.4f}'.format(
                    worker_num, i_episode, self.train_episode, episode_reward,
                    step + 1, end_episode_time
                )
            )
            self.agent.training_info.add_episode_info(end_episode_time, episode_reward)

        self.agent.training_info.end_train()
        print('Worker {} Done'.format(worker_num))


    def learner_process(self):
        print('Learner is ready!')
        save_count = 0
        while self.agent.training_info.num_end_workers() < self.num_workers:  # learn if any worker is still working
            if self.agent.buffer.len() > self.batch_size:
                self.agent.update()
                if self.save_weights:
                    save_count += 1
                    if save_count % 10 == 0:
                        self.agent.save(self.date_time)

    def train_loop(self):
        print('Running the training!!')
        p = Pool(processes=self.num_workers + 1)
        p.apply_async(self.learner_process)
        p.apply_async(self.worker_process, (self.render, 0))
        for i in range(self.num_workers - 1):
            p.apply_async(self.worker_process, (True, i + 1))  # set it to False if you only want to render 1 env
        p.close()
        p.join()

    def test_loop(self):
        env = gym.make(self.env_id)
        start = time.time()
        print('Running the testing!!')
        for i_episode in range(self.test_episode):
            state = env.reset().astype(np.float32)
            episode_reward = 0
            for step in range(self.max_steps):
                env.render()
                state, reward, done, info = env.step(self.agent.policy_net.get_action(torch.Tensor(state).unsqueeze(0).to(self.device), greedy=True))
                state = state.astype(np.float32)
                episode_reward += reward
                if done:
                    break
            print(
                'Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Episode step: {:.4f}  | Running Time: {:.4f}'.format(
                    i_episode, self.test_episode, episode_reward,
                    step, time.time() - start
                )
            )

    def run(self):
        if self.load_weights:
            self.agent.load_weights()
            print('Weight Loaded')

        try:
            if self.if_train:
                self.train_loop()
            self.test_loop()

        except KeyboardInterrupt:
            print('Stopped by User')


class SingleStepRunner:
    def __init__(self, env, env_id, agent, render, batch_size, train_episode, test_episode, up_itr=1,
                 warm_up_steps=0, max_steps=1000000, save_weights=False, load_weights=False, if_train=True, device='cpu'):
        self.env = env
        self.env_id = env_id
        self.agent = agent
        self.render = render
        self.batch_size = batch_size
        self.episode = train_episode
        self.train_episode = train_episode
        self.test_episode = test_episode

        self.up_itr = up_itr
        self.warm_up_steps = warm_up_steps
        self.max_steps = max_steps
        self.save_weights = save_weights
        self.if_train = if_train
        self.load_weights = load_weights
        self.device = device

        self.date_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    def train_loop(self, render):
        print('Running the training!!')
        policy_net = deepcopy(self.agent.policy_net).to('cpu')
        frame_idx = 0
        print('Actor is ready!')
        self.agent.training_info.add_start_time(time.time())
        for i_episode in range(1, self.train_episode, 1):
            state = self.env.reset().astype(np.float32)
            episode_reward = 0
            for step in range(self.max_steps):
                if render:
                    self.env.render()
                if frame_idx >= int (self.warm_up_steps):
                    action = policy_net.get_action(torch.Tensor(state).unsqueeze(0).to('cpu'))
                else:
                    action = policy_net.sample_action()

                next_state, reward, done, _ = self.env.step(action)
                next_state = next_state.astype(np.float32)
                mask = 0.0 if done else self.agent.gamma  # mask = (1 - done) * gamma
                transition = (state, action, reward, next_state, mask)
                self.agent.buffer.push(transition)
                state = next_state
                episode_reward += reward
                frame_idx += 1
                if self.agent.buffer.len() > self.batch_size:
                    self.agent.update()
                    if self.save_weights:
                        self.agent.save(self.date_time)
                policy_net.load_state_dict(self.agent.policy_net.state_dict())

                if done:
                    break

            end_episode_time = time.time() - self.agent.training_info.get_start_time()
            print(
                'Training  | Episode: {}/{}  | Episode Reward: {:.4f}  | Episode step: {:.4f}  | Running Time: {:.4f}'.format(
                    i_episode, self.train_episode, episode_reward,
                    step + 1, end_episode_time
                )
            )
            self.agent.training_info.add_episode_info(end_episode_time, episode_reward)
            self.agent.training_info.plot_episode_reward(self.date_time, self.agent.name, self.env_id)

    def test_loop(self):
        env = gym.make(self.env_id)
        start = time.time()
        print('Running the testing!!')
        for i_episode in range(self.test_episode):
            state = env.reset().astype(np.float32)
            episode_reward = 0
            for step in range(self.max_steps):
                env.render()
                state, reward, done, info = env.step(self.agent.policy_net.get_action(torch.Tensor(state).unsqueeze(0).to(self.device), greedy=True))
                state = state.astype(np.float32)
                episode_reward += reward
                if done:
                    break
            print(
                'Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Episode step: {:.4f}  | Running Time: {:.4f}'.format(
                    i_episode, self.test_episode, episode_reward,
                    step, time.time() - start
                )
            )

    def run(self):
        if self.load_weights:
            self.agent.load_weights()
            print('Weight Loaded')

        try:
            if self.if_train:
                self.train_loop(self.render)
            self.test_loop()

        except KeyboardInterrupt:
            print('Stopped by User')