import numpy as np
import random
import keras
from time import time
import pandas as pd

class DQNAgent:
    def __init__(self, env, model):
        self.env = env
        self.action_size = env.action_space.n
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0    # exploration rate
        self.epsilon_min = 0.075
        self.epsilon_decay = 0
        self.learning_rate = 0.01
        self.model = model
        self.target_network = None
        self.data = []
        self.epsilon_decay_func_dict = {"exponential" : self.epsilon_exponential_decay,
                                        "linear" : self.epsilon_linear_decay,
                                        "constant" : self.epsilon_constant_decay,
                                       }

    def epsilon_exponential_decay(self, epsilon):
         return epsilon * self.epsilon_decay
    
    def epsilon_linear_decay(self, epsilon):
         return epsilon - self.epsilon_decay

    def epsilon_constant_decay(self, epsilon):
         return epsilon

    def _clone_model(self):
        self.target_network = keras.models.clone_model(self.model)
        self.target_network.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        if self.target_network is None:
            act_values=self.model.predict(state)
        else:
            act_values=self.target_network.predict(state)
        return np.argmax(act_values[0])  # returns action

    def act_greedy(self, state): 
        if self.target_network is None:
            act_values=self.model.predict(state)
        else:
            act_values=self.target_network.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch=random.sample(self.memory, batch_size)
        X = []
        y = []

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              index = np.argmax(self.model.predict(next_state)[0])
              target = reward + self.gamma * \
                       self.target_network.predict(next_state)[0][index]

            target_f = self.model.predict(state)
            target_f[0][action] = target
            X.append(state)
            y.append(target_f[0])

        self.model.fit(np.array(X), np.array(y), epochs=1, verbose=0)

    def decay_epsilon(self, epsilon_decay_func):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon_decay_func_dict[epsilon_decay_func](self.epsilon)

    def get_epsilon(self, episodes, epsilon_decay_func):
        if epsilon_decay_func == "exponential":
            return np.exp(np.log(self.epsilon_min)/episodes)
        
        elif epsilon_decay_func == "linear":
            return (1 - self.epsilon_min) / (0.1 * episodes)
        
        elif epsilon_decay_func == "constant":
            return 0

    def train(self, episodes=5000, start_mem=10000, batch_size=24, verbose_eval=1000, save_iter=1000, epsilon_decay_func="exponential", load_target_iter=2500):
        self.get_epsilon(episodes, epsilon_decay_func)
        time_begin = time()
        time_prev = time()

        for e in range(1, episodes + 1):
            state = self.env.reset()
            for time_t in range(400):
                state = self.env.getCurrentState()
                action = self.act(self.env.state)
                next_state, reward, done = self.env.step(action)
                next_state = self.env.getCurrentState()
                self.remember(state, action, reward, next_state, done)
                state = next_state

                if self.env.done:
                    break
            
            if len(self.memory)>start_mem:
                self.replay(batch_size)
            self.decay_epsilon(epsilon_decay_func)

            if e%load_target_iter == 0 and e is not 0:
            	self._clone_model()

        df = pd.DataFrame(self.data, columns=['Episodes', 'Scores', 'Time', 'Best Score', 'Best Time'])
        df.to_csv("trained_models/scores.csv")
        self.model.save("trained_models/final_model.h5")

