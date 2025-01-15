from ConAgent import ConAgent, gas_brake_map
import gym
import numpy as np
from collections import deque
import pandas as pd
import os
import gym_multi_car_racing
import time

env = gym.make("MultiCarRacing-v0", num_agents=1, direction='CCW',
                use_random_direction=True, backwards_flag=True, h_ratio=0.25,
                use_ego_color=False)
start = time.time()
print("Current directory: ", os.getcwd())

path = "Car_ep:20_exp:0.pt"
# path = "Trained_models/Car_LSTM_ep:1000_exp:0.txt"

agent = ConAgent()
agent.load(path)
state = env.reset()

score = 0
step = 0
while True:
    action = agent.sample_action(state)
    next_state, reward, _, _ = env.step(gas_brake_map(action))
    terminated = env.terminated
    score += reward
    step += 1
    state = next_state
    env.render("human")
    if env.terminated:
        break

print("Score: ", score)
print("Time: ", time.time() - start)
env.close()
