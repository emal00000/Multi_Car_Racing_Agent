from ConAgent import traning_loop
import gym
import os
import gym_multi_car_racing.multi_car_racing
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
SAC_exp = traning_loop(env, 
                        num_episodes=7_000,
                        num_experiments=3,
                        min_alpha=0.1,
                        tau=0.1**(1/2_000),
                        render_mode=None,
                        max_steps = 1_200,
                        # load_pretrained="Car_exp:0_ep:3000.pth"
                        )
SAC_exp.start_exp_loop()
t = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
print(f"Time: {t}")
env.close()
