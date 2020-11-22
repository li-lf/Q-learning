import numpy as np
import random
from collections import defaultdict
from abc import abstractmethod


class QAgent:
    def __init__(self, grid_num):
        self.actions = [0, 1, 2, 3]
        self.grid_num = grid_num
        self.lr = 0.1
        self.discount = 0.9
        # self.epsilon = 0.1
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        pass

    def learn(self, ob, action, reward, next_ob):
        state = self.hash_state(ob)
        next_state = self.hash_state(next_ob)
        now_q = self.q_table[state][action]
        new_q = reward + self.discount * max(self.q_table[next_state])
        self.q_table[state][action] += self.lr * (new_q - now_q)

    @abstractmethod
    def select_action(self, ob):
        state = self.hash_state(ob)
        # if np.random.rand() < self.epsilon:
        #     action = np.random.choice(self.actions)
        # else:
        #     state_action = self.q_table[state]
        #     action = self.argmax(state_action)
        state_action = self.q_table[state]
        action = self.argmax(state_action)
        return action

    def hash_state(self, ob):
        return int(ob[0] * self.grid_num + ob[1])

    @staticmethod
    def argmax(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)
