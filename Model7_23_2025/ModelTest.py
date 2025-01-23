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

path = "Checkpoints/Car_exp_0_1000.pth"
# path = "Car_exp:1_ep:5000.pth"


agent = ConAgent()

print(f"Loading pretrained model: {path}")
agent.load(path)
scores = []
for i in range(100):
    state = env.reset()
    score = 0
    step = 0
    while True:
        time.sleep(0.015)
        action = agent.sample_action(state)
        next_state, reward, _, _ = env.step(gas_brake_map(action))
        terminated = env.terminated
        score += reward
        step += 1
        state = next_state
        env.render("human")
        if env.terminated:
            break
    scores.append(score)
    print(f"\nScore: {score[0]:.2f}", )
    print(f"Time: {time.time() - start:.2f}")

print(f"Average score: {np.mean(scores):.2f}")
env.close()
