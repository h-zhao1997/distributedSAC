import torch
import torch.nn as nn
import numpy as np


class QNetwork(nn.Module):
    """ the network for evaluate values of state-action pairs: Q(s,a) """

    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()

        self.nn_layer = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).float()

    def forward(self, states_actions):
        return self.nn_layer(states_actions)


class VNetwork(nn.Module):
    """ the network for evaluate values of state: V(s) """

    def __init__(self, state_dim):
        super(VNetwork, self).__init__()

        self.nn_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).float()

    def forward(self, states_actions):
        return self.nn_layer(states_actions)


class PPOPolicyNetwork(nn.Module):
    def __init__(
            self, state_dim, action_dim, action_range=1.
    ):
        super(PPOPolicyNetwork, self).__init__()

        self.action_range = action_range
        self.action_dim = action_dim

        self.nn_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        ).float()

        self.log_std = nn.parameter.Parameter(
            torch.zeros(action_dim)
        )

    def forward(self, states):
        mean = self.nn_layer(states)
        log_std = self.log_std
        return mean, log_std

    def get_action(self, state, greedy=False):
        mean, log_std = self.forward(state)
        if greedy:
            action = self.action_range * torch.tanh(mean)
        else:
            std = torch.diag_embed(torch.exp(log_std))
            normal = torch.distributions.MultivariateNormal(mean, std)
            action_raw = normal.rsample()
            action_0 = torch.tanh(action_raw)  # TanhNormal distribution as actions; reparameterization trick
            action = self.action_range * action_0

        return action.squeeze(0).detach().cpu().numpy()

    def evaluate(self, state, action, epsilon=1e-6):
        """ generate prob for calculating gradients """
        mean, log_std = self.forward(state)
        std = torch.diag_embed(torch.exp(log_std))  # no clip in evaluation, clip affects gradients flow
        normal = torch.distributions.MultivariateNormal(mean, std)
        action_0 = action / self.action_range
        action_raw = torch.atanh(action_0)
        # according to original paper, with an extra last term for normalizing different action range
        log_prob = normal.log_prob(action_raw) - torch.log(1. - action_0**2 + epsilon).sum(dim=1) - np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action);
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability,
        # needs sum up across the dim of actions to get 1 dim probability; or else use Multivariate Normal.
        return log_prob.unsqueeze(1), mean, log_std



class SACPolicyNetwork(nn.Module):
    """ the network for generating non-deterministic (Gaussian distributed) action from the state input """

    def __init__(
            self, state_dim, action_dim, action_range=1., log_std_min=-20, log_std_max=2
    ):
        super(SACPolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.nn_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        ).float()

        self.actor_layer = nn.Sequential(
            nn.Linear(32, action_dim),
            nn.Tanh()
        ).float()

        self.actor_std_layer = nn.Sequential(
            nn.Linear(32, action_dim),
        ).float()

        self.action_range = action_range
        self.action_dim = action_dim

    def forward(self, states):
        x = self.nn_layer(states)
        mean = self.actor_layer(x[:, :32])
        log_std = self.actor_std_layer(x[:, 32:64])
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def get_action(self, state, greedy=False):
        mean, log_std = self.forward(state)
        if greedy:
            action = self.action_range * torch.tanh(mean)
        else:
            std = torch.diag_embed(torch.exp(log_std))
            normal = torch.distributions.MultivariateNormal(mean, std)
            action_raw = normal.rsample()
            action_0 = torch.tanh(action_raw)  # TanhNormal distribution as actions; reparameterization trick
            action = self.action_range * action_0

        return action.squeeze(0).detach().cpu().numpy()

    def evaluate(self, state, epsilon=1e-6):
        """ generate action with state for calculating gradients """
        mean, log_std = self.forward(state)
        std = torch.diag_embed(torch.exp(log_std))  # no clip in evaluation, clip affects gradients flow
        normal = torch.distributions.MultivariateNormal(mean, std)
        action_raw = normal.rsample()
        action_0 = torch.tanh(action_raw)  # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range * action_0
        # according to original paper, with an extra last term for normalizing different action range
        log_prob = normal.log_prob(action_raw) - torch.log(1. - action_0**2 + epsilon).sum(dim=1) - np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action);
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability,
        # needs sum up across the dim of actions to get 1 dim probability; or else use Multivariate Normal.
        return action, log_prob.unsqueeze(1), mean, log_std

    def sample_action(self):
        """ generate random actions for exploration """
        action_0 = torch.rand(self.action_dim) * 2 - 1
        action = self.action_range * action_0
        return action.numpy()