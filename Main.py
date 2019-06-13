import os
import sys

module_path = os.path.abspath(os.path.join('C:\\Users\\r.hakimov\\Documents\\GitHub\\ReinforcementLearning'))
if module_path not in sys.path:
    sys.path.append(module_path)

from Models.A2C import A2C
from CustomEnvironments.DimensionWalk import DimensionWalk
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from Utils import encode_actions
NUM_EPOCHS = 15


def run_training(env, model, action_array):
    actor_losses = []
    critic_losses = []
    total_rewards = []
    for i_episode in range(NUM_EPOCHS):
        if i_episode % 100 == 1:
            print(str(i_episode) + ' Total Reward = ' + str(total_rewards[-1]))
        states, actions, rewards, next_states, dones = [], [], [], [], []
        observation = env.reset()
        done = False
        k = 0
        total_reward = 0
        while done == False:

            k = k + 1
            action = model.predict_action(np.array([observation]))
            states.append(observation)
            observation, reward, done, info = env.step(action_array[action])
            if (k > 200):
                done = True

            next_states.append(observation)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            total_reward += reward
        actor_loss, critic_loss = model.train(states, actions, rewards, next_states, dones)
        total_rewards.append(total_reward)
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
    return total_rewards

env = DimensionWalk([10,10])
encoded_actions = [[0,1],[1,0],[-1,0],[0,-1]]
model = A2C(env.observation_space,len(encoded_actions),[4],lr_actor=0.005,lr_critic=0.005,gamma=0.9)
a2c_total_rewards = run_training(env,model,encoded_actions)