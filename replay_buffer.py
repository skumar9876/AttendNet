import numpy as np
import random


class ReplayBuffer(object):

    def __init__(self, max_size, batch_size):
        self.max_size = max_size
        self.batch_size = batch_size

        self.states = [None] * self.max_size
        self.actions = [None] * self.max_size
        self.rewards = [None] * self.max_size
        self.next_states = [None] * self.max_size
        self.terminals = [None] * self.max_size

        self.curr_pointer = 0
        self.curr_size = 0

    def add(self, state, action, reward, next_state, terminal):
        self.states[self.curr_pointer] = state
        self.actions[self.curr_pointer] = action
        self.rewards[self.curr_pointer] = reward
        self.next_states[self.curr_pointer] = next_state
        self.terminals[self.curr_pointer] = terminal

        self.curr_pointer += 1
        self.curr_size = min(self.max_size, self.curr_size + 1)
        # If replay buffer is full, set current pointer to be at the beginning of the buffer.
        if self.curr_pointer >= self.max_size:
            self.curr_pointer -= self.max_size

    def sample(self):
        if self.curr_size < self.batch_size:
            return [], [], [], [], []
        sample_indices = []

        # Ensure that the most recent transition is in the returned batch.
        sample_indices.append(self._curr_pointer - 1)
        for i in xrange(self.batch_size - 1):
            sample_indices.append(random.randint(0, self.curr_size - 1))

        return self.states[sample_indices], self.actions[sample_indices], self.rewards[sample_indices], self.next_states[sample_indices], self.terminals[sample_indices]