import numpy as np
import random

class QLearningAgent(object):

    DISCOUNT = 0.95

    def __init__(self, num_states, num_actions, learning_rate, epsilon):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.q_table = np.zeros((self.num_states, self.num_actions))
        self.curr_transition = None

    def sample(self, state):
        state_index = list(state).index(1)
        q_values = self.q_table[state_index]

        e = random.random()
        if e < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(q_values)

    def best_action(self, state):
        state_index = np.where(np.squeeze(state) == 1)[0]
        q_values = self.q_table[state_index]
        return np.argmax(q_values)

    def store(self, state, action, reward, next_state, terminal, eval):
        if not eval:
            self.curr_transition = [state, action, reward, next_state, terminal]

    def update(self):
        state = self.curr_transition[0]
        action = self.curr_transition[1]
        reward = self.curr_transition[2]
        next_state = self.curr_transition[3]
        terminal = self.curr_transition[4]

        state_index = np.where(np.squeeze(state)==1)[0]
        next_state_index = np.where(np.squeeze(next_state)==1)[0]

        td_target = reward + (1 - terminal) * self.DISCOUNT * np.max(self.q_table[next_state_index])

        self.q_table[state_index, action] = (
            1 - self.learning_rate) * self.q_table[state_index, action] + self.learning_rate * td_target