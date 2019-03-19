from Models.A2C import A2C
import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

NUM_EPOCHS = 2500


def run_training(env, model):
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
            observation, reward, done, info = env.step(action)
            reward += k * 0.001
            if (done == True):
                reward = -10
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

env = gym.make('BipedalWalker-v2')
model = A2C(env.observation_space.shape[0],2,[8,16],lr_actor=0.001,lr_critic=0.001,gamma=0.99)
a2c_total_rewards = run_training(env,model)

running_rewards = []
running_reward = 0
for i in range(len(a2c_total_rewards)):
    running_reward = 0.9*running_reward+0.1*a2c_total_rewards[i]
    running_rewards.append(running_reward)
plt.plot(range(0,NUM_EPOCHS),running_rewards)