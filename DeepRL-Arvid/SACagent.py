import torch
import torch.nn as nn
import numpy as np

import random
from collections import namedtuple, deque

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "terminated")
)
Batch = namedtuple(
    "Batch", ("states", "actions", "rewards", "next_states", "terminateds")
)


def state_encoder(state: torch.Tensor):
    encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
    if len(state.shape) == 3:
        state = state.unsqueeze(1)
    elif len(state.shape) == 2:
        state = state.unsqueeze(0).unsqueeze(1)

    x = encoder(state[:,:, :-1, :]).squeeze(0) # encoding track info
    car_info = state[:, :, -1, :].squeeze((0,1))
    return torch.cat((x, car_info), dim=-1)



class ReplayMemory:

    def __init__(self, capacity: int, batch_size: int):
        self.batch_size = batch_size
        self.data = deque([], maxlen=capacity)

    def add(self, *args):
        self.data.append(Transition(*args))

    def sample(self) -> Batch:
        sample = random.choices(self.data, k=self.batch_size)
        states, actions, rewards, next_states, terminateds = list(zip(*sample))
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        terminateds = torch.tensor(np.array(terminateds), dtype=torch.float32)
        return Batch(states, actions, rewards, next_states, terminateds)

    def __len__(self):
        return len(self.data)


class Qfunction(nn.Module):
    """
    SAC uses Q(s,a) instead of DQN's Q(s, :)
    """

    def __init__(self):
        super().__init__()
        self.state_encoder = state_encoder
        self.dense_net = nn.Sequential(  
                nn.Linear(518, 600),
                nn.ReLU(),
                nn.Linear(600, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
        )

    def forward(self, states, actions):
        embedding = self.state_encoder(states)
        x = torch.cat((embedding, actions.view(actions.shape[0], -1)), dim=1)
        x = self.dense_net(x)
        return x
        


class GaussianPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = state_encoder
        self.fc = nn.Linear(515, 600)
        self.mu = nn.Sequential(
            nn.ReLU(),
            nn.Linear(600, 3)
        )
        self.std = nn.Sequential(
            nn.ReLU(),
            nn.Linear(600, 3)
        )

    def forward(self, state):
        # Reparameterized and squashed sampling 
        # Returns actions and log probabilities
        encoding = self.encoder(state)
        encoding = self.fc(encoding)
        gaussian = torch.distributions.Normal(self.mu(encoding), torch.abs(self.std(encoding)))
        u = gaussian.rsample()
        a = torch.tanh(u)
        
        log_pi_u = gaussian.log_prob(u).sum(axis=-1)
        inv_det_jacobian = torch.sum(
            2 * (np.log(2) - u - nn.functional.softplus(-2 * u)), dim=-1
        )
        log_prob_a = log_pi_u - inv_det_jacobian
        return a, log_prob_a


class SacAgent(nn.Module):
    """
    SAC agent: 
        Policy, two Q, two target Qs
        Three optimizers
        Memory
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.lambda_ = 0.005
        self.gamma = 0.99
        self.alpha = 1.0

        self.memory = ReplayMemory(capacity=100_000, batch_size=64)
        self.pi = GaussianPolicy()
        self.qs = [Qfunction(), Qfunction()]
        self.tqs = [Qfunction(), Qfunction()]

        self.optimizer_pi = torch.optim.Adam(self.pi.parameters(), lr=1e-3)
        self.q_optimizers = [torch.optim.Adam(self.qs[0].parameters(), lr=1e-3),
                       torch.optim.Adam(self.qs[1].parameters(), lr=1e-3)]
        self.clone()


    @torch.no_grad
    def clone(self):
        # Hard update (for initial copy)
        self.tqs[0].load_state_dict(self.qs[0].state_dict())
        self.tqs[1].load_state_dict(self.qs[1].state_dict())

    @torch.no_grad
    def soft_update(self):
        # Soft update of targets
        # Network 1
        for target_param, Qnet_param in zip(self.tqs[0].parameters(), self.qs[0].parameters()):
            target_param.data.copy_((1.0 - self.lambda_) * target_param.data + self.lambda_ * Qnet_param.data)
        # Network 2
        for target_param, Qnet_param in zip(self.tqs[1].parameters(), self.qs[1].parameters()):
            target_param.data.copy_((1.0 - self.lambda_) * target_param.data + self.lambda_ * Qnet_param.data)

    @torch.no_grad
    def sample_action(self, state):
        # Sample single action from the policy
        action, _ = self.pi(state)
        action[1:3] = 0.5 * action[1:3] + 0.5 # normalazing gas and braking to [0, 1] 
        return action

    def update(self, batch: Batch):
        # Update all parameters given batch
        states, actions, rewards, next_states, terminateds = batch

        # Update Qs
        q_est_0 = self.qs[0](states, actions)
        q_est_1 = self.qs[1](states, actions)
        with torch.no_grad():
            next_actions, next_log_probs = self.pi(next_states)
            next_q_0 = self.tqs[0](next_states, next_actions)
            next_q_1 = self.tqs[1](next_states, next_actions)
            min_q = torch.min(next_q_0, next_q_1)
            target = rewards.reshape(-1, 1) + self.gamma * (1 - terminateds.reshape(-1, 1)) * (
                min_q - self.alpha * next_log_probs.reshape(-1, 1)
            )
        for q_est, opt in zip([q_est_0, q_est_1], self.q_optimizers):
            q_loss = nn.functional.mse_loss(q_est, target)
            q_loss.backward()
            opt.step()
            opt.zero_grad()

        # Update policy
        # Freeze qs for efficiency
        self.qs[0].requires_grad_(False)
        self.qs[1].requires_grad_(False)

        acts, log_probs = self.pi(states) # With states sampled from buffer
        min_q = torch.min(self.qs[0](states, acts), self.qs[1](states, acts))
        pi_loss = torch.mean(self.alpha * log_probs - min_q)
        pi_loss.backward()

        self.optimizer_pi.step()
        self.optimizer_pi.zero_grad()

        # Unfreeze qs again
        self.qs[0].requires_grad_(True)
        self.qs[1].requires_grad_(True)


    def store(self, state, action, reward, next_state, terminated):
        # Store transition
        self.memory.add(state, action, reward, next_state, terminated)

    def learn(self):
        # Just to call agent.learn() from some loop
        batch = self.memory.sample()
        self.update(batch)

    def save(self, filename: str):
        torch.save(self.pi.state_dict(), filename)

    def load(self, filename: str):
        self.pi.load_state_dict(torch.load(filename, weights_only=True))