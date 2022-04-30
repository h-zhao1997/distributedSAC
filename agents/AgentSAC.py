import torch
import copy
import torch.nn as nn
from torch.optim.adam import Adam
import numpy as np
from distributedSAC.agents.net import *
from distributedSAC.train.utils import *
from distributedSAC.train.buffer import *
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class AgentSAC:
    def __init__(
            self, state_dim, action_dim, action_range, buffer, training_info, batch_size, gamma, env_id,
            auto_entropy=True, target_entropy=2.,
            soft_tau=1e-2, reward_scale=10., soft_q_lr=3e-1, policy_lr=3e-4, alpha_lr=3e-4, device='cpu'
    ):
        self.buffer = buffer
        self.training_info = training_info
        self.name = 'SAC'
        self.device = device
        self.soft_tau = soft_tau
        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.auto_entropy = auto_entropy
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.env_id = env_id

        # initialize all networks
        self.soft_q_net1 = QNetwork(state_dim, action_dim).to(device)
        self.soft_q_net2 = QNetwork(state_dim, action_dim).to(device)
        self.policy_net = SACPolicyNetwork(state_dim, action_dim, action_range=action_range).to(device)
        self.soft_q_net1.train()
        self.soft_q_net2.train()
        self.policy_net.train()

        # initialize weights of target networks
        self.target_soft_q_net1 = copy.deepcopy(self.soft_q_net1).to(device)
        self.target_soft_q_net2 = copy.deepcopy(self.soft_q_net2).to(device)
        self.target_soft_q_net1.eval()
        self.target_soft_q_net2.eval()

        self.log_alpha = torch.nn.Parameter(data=torch.Tensor([1.]).to(device), requires_grad=True)
        if auto_entropy:
            self.alpha = torch.exp(self.log_alpha).detach()
        else:
            self.alpha = torch.Tensor([1.]).detach().to(device)

        self.soft_q_optimizer1 = Adam(list(self.soft_q_net1.parameters()), lr=soft_q_lr)
        self.soft_q_optimizer2 = Adam(list(self.soft_q_net2.parameters()), lr=soft_q_lr)
        self.policy_optimizer = Adam(list(self.policy_net.parameters()), lr=policy_lr)
        self.alpha_optimizer = Adam([self.log_alpha], lr=alpha_lr)

    def target_soft_update(self, net, target_net):
        """ soft update the target net with Polyak averaging """
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * self.soft_tau + param.data * (1.0 - self.soft_tau))
        return target_net

    def update(self):
        """ update all networks in SAC """
        state, action, reward, next_state, mask = self.buffer.sample(self.batch_size)

        reward = reward[:, np.newaxis]  # expand dim
        mask = mask[:, np.newaxis]

        reward = self.reward_scale * (reward - np.mean(reward, axis=0)) / (
                 np.std(reward, axis=0) + 1e-6
        )  # normalize with batch mean and std; plus a small number to prevent numerical problem

        # Training Q Function
        new_next_action, next_log_prob, _, _ = self.policy_net.evaluate(torch.Tensor(next_state).to(self.device))
        target_q_input = torch.cat([torch.Tensor(next_state).to(self.device), new_next_action], 1)  # the dim 0 is number of samples
        target_q_min = torch.min(
            self.target_soft_q_net1(target_q_input), self.target_soft_q_net2(target_q_input)
        ) - self.alpha * next_log_prob
        target_q_value = torch.Tensor(reward).to(self.device) + torch.Tensor(mask).to(self.device) * target_q_min  # only reward if done==1 (mask == 0)
        target_q_value = target_q_value.detach()
        q_input = torch.cat([torch.Tensor(state).to(self.device), torch.Tensor(action).to(self.device)], 1)
        value_loss = nn.MSELoss()

        self.soft_q_optimizer1.zero_grad()
        predicted_q_value1 = self.soft_q_net1(q_input)
        q_value_loss1 = value_loss(predicted_q_value1, target_q_value)
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()

        self.soft_q_optimizer2.zero_grad()
        predicted_q_value2 = self.soft_q_net2(q_input)
        q_value_loss2 = value_loss(predicted_q_value2, target_q_value)
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

        # Training Policy Function
        self.policy_net.zero_grad()
        new_action, log_prob, mean, log_std = self.policy_net.evaluate(torch.Tensor(next_state).to(self.device))
        new_q_input = torch.cat([torch.Tensor(state).to(self.device), new_action], 1)  # the dim 0 is number of samples
        predicted_new_q_value = torch.min(self.soft_q_net1(new_q_input), self.soft_q_net2(new_q_input))
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Updating alpha w.r.t entropy
        # alpha: trade-off between exploration (max entropy) and exploitation (max Q)
        if self.auto_entropy:
            self.alpha_optimizer.zero_grad()
            alpha_loss = -((self.log_alpha * (log_prob.detach() + self.target_entropy))).mean()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = torch.exp(self.log_alpha).detach()

        # Soft update the target value nets
        self.target_soft_q_net1 = self.target_soft_update(self.soft_q_net1, self.target_soft_q_net1)
        self.target_soft_q_net2 = self.target_soft_update(self.soft_q_net2, self.target_soft_q_net2)


    def save(self, date_time):
        path = os.path.join('model_and_image', '_'.join([self.name, self.env_id])) + '/' + date_time + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        extend_path = lambda s: os.path.join(path, s)
        torch.save(self.soft_q_net1.state_dict(), extend_path('q_net1_params.pth'))
        torch.save(self.soft_q_net2.state_dict(), extend_path('q_net2_params.pth'))
        torch.save(self.target_soft_q_net1.state_dict(), extend_path('target_q_net1_params.pth'))
        torch.save(self.target_soft_q_net2.state_dict(), extend_path('target_q_net2_params.pth'))
        torch.save(self.policy_net.state_dict(), extend_path('policy_net_params.pth'))
        if self.auto_entropy:
            np.save(extend_path('log_alpha.npy'), self.log_alpha.to('cpu').detach().numpy())  # save log_alpha variable

    def load_weights(self):  # load trained weights
        path = os.path.join('model', '_'.join([self.name, self.env_id]))
        extend_path = lambda s: os.path.join(path, s)
        self.soft_q_net1.load_state_dict(torch.load(extend_path('q_net1_params.pth')))
        self.soft_q_net2.load_state_dict(torch.load(extend_path('q_net2_params.pth')))
        self.target_soft_q_net1.load_state_dict(torch.load(extend_path('target_q_net1_params.pth')))
        self.target_soft_q_net2.load_state_dict(torch.load(extend_path('target_q_net2_params.pth')))
        self.policy_net.load_state_dict(torch.load(extend_path('policy_net_params.pth')))
        if self.auto_entropy:
            self.log_alpha.data[0] = torch.Tensor(np.load(extend_path('log_alpha.npy'))).to(self.device)  # load log_alpha variable
