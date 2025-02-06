from ConAgent import ConAgent, gas_brake_map
import gym
import numpy as np
from collections import deque
import pandas as pd
import os
import gym_multi_car_racing
import time

env = gym.make("MultiCarRacing-v0", 
               num_agents=1,
               direction='CCW',
               use_random_direction=True,
               backwards_flag=True,
               h_ratio=0.25,
               use_ego_color=False,
               num_previous_states=3,
               verbose=0
               )
start = time.time()
print("Current directory: ", os.getcwd())

path = "Model/Car_exp:0_ep:3000.pth"
path = "Model/Car_exp:0_ep:7000.pth"



agent = ConAgent()

print(f"Loading pretrained model: {path}")
agent.load(path)
scores = []
finished = 0
for i in range(100):
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
        if i < 10:
            time.sleep(0.015)
            env.render("human")
        if env.terminated:
            break
    scores.append(score)
    if score > 920:
        finished += 1
    print(f"\nScore: {score[0]:.2f}", )
    print(f"Time: {time.time() - start:.2f}")

print(f"Average score: {np.mean(scores):.2f}")
print(f"Laps completed {finished:d} out of 100")
env.close()
