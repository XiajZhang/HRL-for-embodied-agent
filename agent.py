"""
Define different Agent including
    1. basic Q learning agent
    2. hierarchical Q learning agent with Options framework
"""
from abc import ABC, abstractmethod
from copyreg import pickle
from math import degrees
from pyexpat import model
from tkinter.messagebox import NO
from barl_simpleoptions import State
import numpy as np
from enum import Enum
import random
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
from embodied_env import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import os
import pickle

class Agent(ABC):
    """
    agent need to :
    1. select next action based on current state
    2. update policy based on reward
    """

    def __init__(self, actions, states) -> None:
        pass
    
    @abstractmethod
    def get_next_action(self, current_state : State):
        return -1

    @abstractmethod
    def update_agent(self, reward, next_state : State) -> None:
        pass

class AgentExploration(Enum):
    Eps_Greedy = 0,
    Upper_Conf_Bound = 1

class TabularQAgent(Agent):
    def __init__(self, actions_n, states_n, exploration : AgentExploration, gamma, lr, eps=0.1) -> None:
        # super().__init__(actions, states)
        self.n_state = actions_n
        self.n_action = states_n
        self._Q = np.zeros((self.n_state, self.n_action))
        self._UCB = np.zeros((self.n_state, self.n_action))
        self.gamma = gamma
        self.eps = eps
        self.lr = lr
        self.exploration = exploration
        self.last_state = None
        self.last_action = None

    def get_next_action(self, current_state : State):
        action_idx = np.argmax(self._Q[current_state, :])
        if self.exploration == AgentExploration.Eps_Greedy:
            rdn = random.random()
            if rdn < self.eps:
                action_idx = random.choice(list(range(self.n_action)))
                print("Eps greedy action...")
        elif self.exploration == AgentExploration.Upper_Conf_Bound:
            # Calculate upper confidence bound for exploration
            pass
        print("The action selected is ", action_idx)
        print(self.actions[action_idx])
        self.last_state = current_state
        self.last_action = self.actions[action_idx]
        return self.last_action
    
    def update_agent(self, reward, next_state : State) -> None:
        old_q = self._Q[self.last_state][self.last_action]
        self._Q[self.last_state][self.last_action] = old_q + self.lr(reward + self.gamma(np.max(self._Q[next_state])) - old_q)

class RandomAgent(Agent):

    def __init__(self, action_n):
        # super().__init__(actions, states)
        np.random.seed(2)
        self.action_n = action_n

    def get_next_action(self, state):
        selected = np.random.choice(list(range(self.action_n)))
        return selected
    
    def update_agent(self, state_vector, action, next_state_vector, reward) -> None:
        # return super().update_agent(reward, next_state)
        pass


class PolynomialLinearValueApproximationAgent(Agent):

    def __init__(self, actions_n, env, gamma=1.0, epsilon=0.1, learning_rate=0.1, scaler=False) -> None:
        np.random.seed(2)
        self._dir = os.path.join(os.getcwd(), "model")
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions_n = actions_n
        self.lr = learning_rate
        if scaler:
            self.scaler = self.get_scaler(env)
        else:
            self.scaler = None
        # Featurizer
        self.featurizer = PolynomialFeatures(degree=2)
        # Initialize model
        self.models = self.initialize_estimators(actions_n, env.reset())
    
    def save_policy(self):

        if not os.path.exists(self._dir):
            os.mkdir(self._dir)
        
        with open(os.path.join(self._dir, "polynoLVA_agent5.pkl"), 'wb') as f:
            pickle.dump(self, f)            

    def get_scaler(self, env):
        sampled_obs = [] 
        for _ in range(500):
            sampled_obs += env.sample() 
        scaler = StandardScaler()
        scaler.fit(sampled_obs)
        return scaler

    def state_featurizer(self, state_vector):
        """
            Receive a state vector and turn them into a three polynomial feature vector
        """
        if self.scaler is not None:
            scaled = self.scaler.transform(state_vector)
        else:
            scaled = state_vector
        features = self.featurizer.fit_transform(scaled)
        # print()
        # if features.shape[0] == 1:
        #     features = features.squeeze(0)
        # print(features.shape)
        return features

    def initialize_estimators(self, actions_n, initial_state):
        # TODO: Try to load pickle file first
        models = []
        for _ in range(actions_n):
            _mod = SGDRegressor(learning_rate="constant",eta0=self.lr).partial_fit(self.state_featurizer([initial_state]), [0])
            models.append(_mod)
        return models

    def get_next_action(self, state_vector):
        # Featurize
        features = self.state_featurizer([state_vector])
        # Prediction
        preds = np.array([_mod.predict(features)[0] for _mod in self.models])
        best_action = np.argmax(preds)

        action_probs = np.ones(self.actions_n, dtype=float) * self.epsilon/self.actions_n
        action_probs[best_action] += 1 - self.epsilon
        selected_action = np.random.choice(range(self.actions_n), p=action_probs)
        return selected_action

    def update_agent(self, state_vector, action, next_state_vector, reward) -> None:

        current_state_features = self.state_featurizer([state_vector])
        next_state_features = self.state_featurizer([next_state_vector])
        td = reward + self.gamma * np.max(np.array([_mod.predict(next_state_features)[0] for _mod in self.models]))
        # Update model
        self.models[action].partial_fit(current_state_features, [td])

    def get_q_value(self, state_vector, action_idx):
        features = self.state_featurizer([state_vector])
        var = self.models[action_idx].predict(features)[0]
        return var


    def batch_update(self, mini_batch):
        last_obs, next_obs, rewards = [[], [], [], []], [[], [], [], []], [[], [], [], []]
        for instance in mini_batch:
            index = instance[1]
            last_obs[index].append(instance[0])
            next_obs[index].append(instance[2])
            rewards[index].append(instance[3])
        
        action_cnt = []
        for action in range(self.actions_n):
            action_cnt.append(len(last_obs[action]))
            if len(last_obs[action]) == 0:
                continue
            # print("Action %d happened %d"%(action, len(last_obs[action])))
            current_state_features = self.state_featurizer(last_obs[action])
            next_state_features = self.state_featurizer(next_obs[action])
            td = np.array(rewards[action]) + self.gamma*np.max(np.array([_mod.predict(next_state_features) for _mod in self.models]))
            # print(td)
            # print(td.shape)
            self.models[action].partial_fit(current_state_features, td)

        return action_cnt
