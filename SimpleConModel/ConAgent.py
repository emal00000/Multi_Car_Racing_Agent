import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import itertools
import random
from collections import namedtuple, deque

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "terminated")
)
Batch = namedtuple(
    "Batch", ("states", "actions", "rewards", "next_states", "terminateds")
)


class state_encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
                nn.Conv2d(4, 8, kernel_size=2, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(8, 4, kernel_size=2, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
                )    


    def forward(self, state):
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        x = self.encoder(state) # encoding track info
        return x



class ReplayMemory:

    def __init__(self, capacity: int, batch_size: int):
        self.batch_size = batch_size
        self.data = deque([], maxlen=capacity)

    def add(self, *args):
        self.data.append(Transition(*args))

    def sample(self) -> Batch:
        sample = random.choices(self.data, k=self.batch_size)
        states, actions, rewards, next_states, terminateds = list(zip(*sample))
        states = torch.tensor(np.array(states), dtype=torch.float32,device=device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32,device=device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32,device=device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32,device=device)
        terminateds = torch.tensor(np.array(terminateds), dtype=torch.float32,device=device)
        return Batch(states, actions, rewards, next_states, terminateds)

    def __len__(self):
        return len(self.data)


class Qfunction(nn.Module):
    """
    SAC uses Q(s,a) instead of DQN's Q(s, :)
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.dense_net = nn.Sequential(  
                nn.Linear(62, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
        )

    def forward(self, states, actions):
        encoding = self.encoder(states)
        x = torch.cat((encoding, actions.view(actions.shape[0], -1)), dim=1)
        x = self.dense_net(x)
        return x
        


class GaussianPolicy(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(60, 32)
        self.mu = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        self.std = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, states):
        # Reparameterized and squashed sampling 
        # Returns actions and log probabilities
        encoding = self.encoder(states)
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


class ConAgent(nn.Module):
    """
    SAC agent: 
        Policy, two Q, two target Qs
        Three optimizers
        Memory
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lambda_ = 0.005
        self.gamma = 0.99
        self.alpha = 1.0

        self.memory = ReplayMemory(capacity=100_000, batch_size=64)
        self.encoder = state_encoder() # Shared encoder

        self.pi = GaussianPolicy(self.encoder)
        self.qs = [Qfunction(self.encoder), Qfunction(self.encoder)]
        self.tqs = [Qfunction(self.encoder), Qfunction(self.encoder)]
        
        self.encoder.to(self.device)
        self.pi.to(self.device)
        self.qs[0].to(self.device)
        self.qs[1].to(self.device)
        self.tqs[0].to(self.device)
        self.tqs[1].to(self.device)



        self.optimizer_encoder = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)
        self.optimizer_pi = torch.optim.Adam(itertools.chain(self.pi.fc.parameters(),self.pi.mu.parameters(),self.pi.parameters()), lr=1e-3)
        self.q_optimizers = [torch.optim.Adam(self.qs[0].dense_net.parameters(), lr=1e-3),
                       torch.optim.Adam(self.qs[1].dense_net.parameters(), lr=1e-3)]
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
        state = state.to(self.device)
        action, _ = self.pi(state) 

        action=action.to("cpu") # Return action to CPU

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

        # Update encoder
        self.optimizer_encoder.step()
        self.optimizer_encoder.zero_grad()

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



def gas_brake_map(action):
    # maping negative gas to brake
    action = action.squeeze(0)
    a = torch.cat([action, -action[1].unsqueeze(0)])
    if action[1] > 0:
        a[2] = 0
    else:
        a[1] = 0

    # mininum stering craiterion
    if abs(a[0]) < 0.1:
        a[0] = 0
    return a.numpy()



class traning_loop():
    

    def __init__(self,
                 env,
                num_episodes: int,
                num_experiments: int,
                min_alpha: float,
                tau: float,
                render_mode=None,
                max_steps = 2000,
                load_pretrained=""
                ):
        
        self.env = env
        self.num_episodes = num_episodes
        self.num_experiments = num_experiments
        self.min_alpha = min_alpha
        self.tau = tau
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.load_pretrained = load_pretrained


    def start_exp_loop(self):
        """
        Initiates the experiment loop and aggregates the results.

        Executes a series of experiments as defined by the parameters and
        processes the resulting data into a single Pandas DataFrame.

        Structure of training_loop:
        start_experiment_loop()
        ├── episode_loop()
        │   ├── step_loop()

        Returns:
            pd.DataFrame: A concatenated DataFrame containing the results
                        of all experiments.
        """
        print("Starting experiment loop...")
        dfs = []
        self.high_score = -1e3
        self.high_score_txt = ""
        for exp in range(self.num_experiments):
            self.agent = ConAgent()
            if self.load_pretrained:
                self.agent.load(self.load_pretrained)
            ma10 = self.episode_loop(exp)
            dfs.append(
                pd.DataFrame(
                    {"exp": exp, "episode": np.arange(self.num_episodes), "MA10": ma10}
                )
            )
            self.agent.save(f"Car_ep:{self.num_episodes}_exp:{exp}.pt")

        df = pd.concat(dfs, ignore_index=True)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        df.to_csv(path_or_buf=os.path.join(script_dir, f"ma10_result_exp{exp}.csv"), sep=',', index=False, header=True)
        print()
        return df


    def episode_loop(self, exp):
        scores = deque([], maxlen=10)
        ma10 = np.zeros(self.num_episodes)
        for e in range(self.num_episodes):
            score, step = self.step_loop()
            self.display(exp, e, score, step)
            self.agent.alpha = max(self.min_alpha, self.agent.alpha * self.tau)
            scores.append(score)
            ma10[e] += np.mean(scores)
            if e % 50 == 0:
                print("Checkpoint saved at episode: ", e, "\n")
                self.agent.save(f"Checkpoints/Car_lstm_exp_{exp}_{e}.pt")
        return ma10


    def step_loop(self):
        state = self.env.reset()
        terminated = False
        score = 0
        step = 0
        while not (terminated):
            action = self.agent.sample_action(state)
            next_state, reward, _, _ = self.env.step(gas_brake_map(action))
            terminated = self.env.terminated
            self.agent.store(state, action, reward, next_state, terminated)
            self.agent.learn()
            self.agent.soft_update()
            score += reward
            step += 1
            state = next_state
            if self.render_mode is not None:
                self.env.render(mode=self.render_mode)
            if step >= self.max_steps:
                terminated = True

        return score, step


    def display(self, exp, e, score, step):
        text = f"Experiment: {exp}, Episode: {e}, Score: {score[0]:.2f}, Steps: {step}"
        print(" " * (len(text) + 5), end="\r")
        print(text, end="\r")
        with open("Traning_Log.txt", "w") as f:
            if score[0] > self.high_score:
                self.high_score = score[0]
                self.high_score_txt = text
            f.write("High score:\n"+self.high_score_txt+"\n\nCurrent run:\n"+text)