from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np

class A2C:
    def __init__(self,num_inputs,num_outputs,hidden_layers,lr_actor = 0.001,lr_critic = 0.001, gamma = 0.99):
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_layers = hidden_layers
        self.actor = self.build_actor()
        self.critic = self.build_critic()

    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(self.num_inputs,activation='relu'))
        for hidden in self.hidden_layers:
            actor.add(Dense(hidden,activation='relu'))
        actor.add(Dense(self.num_outputs,activation='softmax'))
        actor.compile(loss='categorical_crossentropy',optimizer=Adam(self.lr_actor))
        return actor

    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(self.num_inputs,activation='relu'))
        for hidden in self.hidden_layers:
            critic.add(Dense(hidden,activation='relu'))
        critic.add(Dense(1,activation='linear'))

        critic.compile(loss='mse',optimizer=Adam(self.lr_critic))
        return critic

    def predict_action(self,state):
        policy = self.actor.predict(state)
        return np.random.choice(self.num_outputs,1,p=policy[0])[0]

    def predict_state_value(self,state):
        state_value = self.critic.predict(state)
        return state_value

    def train(self,states,actions,rewards,next_states,dones):
        values = self.predict_state_value(np.array(states))
        advantages = np.zeros((len(states), self.num_outputs))
        target = np.array(rewards)
        next_value = 0
        for i in reversed(range(len(rewards))):
            if dones[i]:
                advantages[i][actions[i]] = rewards[i] - values[i]
                target[i] = rewards[i]
                next_value = rewards[i] - values[i]
            else:
                advantages[i][actions[i]] = rewards[i] + self.gamma * next_value
                target[i] = rewards[i] + self.gamma * next_value
                next_value = advantages[i][actions[i]]

        actor_loss = self.actor.fit(np.array(states),np.array(advantages),verbose=0)
        critic_loss = self.critic.fit(np.array(states), np.array(rewards),verbose=0)

        return actor_loss.history['loss'], critic_loss.history['loss']

    def discount_rewards(self,rewards,dones):
        discounted_rewards = np.array([])
        cummulative_reward = 0
        for i in range(0,len(rewards)):
            if (dones[i]!=True):
                cummulative_reward = rewards[i]+self.gamma*cummulative_reward
            else:
                cummulative_reward = rewards[i]
            discounted_rewards = np.append(discounted_rewards,[cummulative_reward])
        return discounted_rewards