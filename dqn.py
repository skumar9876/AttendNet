import tensorflow as tf
from replay_buffer import ReplayBuffer
import random

class DqnAgent(object):

    # Discount factor for future rewards.
    DISCOUNT = 0.95
    # Max size of the replay buffer.
    REPLAY_MEMORY_SIZE = 500000
    # Batch size for updates from the replay buffer.
    BATCH_SIZE = 32
    # Update the target network every TARGET_UPDATE timesteps.
    TARGET_UPDATE = 10000

    def __init__(self, sess=None, learning_rate=0.00025, state_dims=[], num_actions=0,
        epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=50000):

        self._learning_rate = learning_rate
        self._state_dims = state_dims
        self._num_actions = num_actions

        self._epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
        self._epsilon_decay_steps = epsilon_decay_steps

        self._replay_buffer = ReplayBuffer(replay_memory_size, batch_size)

        self._current_time_step = 0

        with tf.Graph().as_default():
            self._construct_graph()
            self._saver = tf.train.Saver()
            if sess is None:
                self.sess = tf.Session()
            else:
                self.sess = sess
            self.sess.run(tf.global_variables_initializer())

    def _q_network(self, state):
        # Three convolutional layers
        conv1 = tf.contrib.layers.conv2d(
            state, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(
            conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(
            conv2, 64, 3, 1, activation_fn=tf.nn.relu)

        # Fully connected layers
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        q_values = tf.contrib.layers.fully_connected(fc1, self._num_actions)

        return q_values

    def _construct_graph(self):
        shape=[None]
        for dim in self.state_dims:
            shape.append(dim)
        self._state = tf.placeholder(shape=shape, dtype=tf.float32)

        with tf.variable_scope('q_network'):
            self._q_values = self._q_network(self._state)
        with tf.variable_scope('target_q_network'):
            self._target_q_values = self._q_network(self._state)
        with tf.variable_scope('q_network_update'):
            self._picked_actions = tf.placeholder(shape=[None, 2], dtype=tf.int32)
            self._td_targets = tf.placeholder(shape=[None], dtype=tf.float32)
            self._q_values_pred = tf.gather_nd(self._q_values, self._picked_actions)
            self._losses = tf.squared_difference(self._q_values_pred, self._td_targets)
            self._loss = tf.reduce_mean(self._losses)

            self.optimizer = tf.train.RMSPropOptimizer(self._learning_rate, 0.99, 0.0, 1e-6)
            self.train_op = self.optimizer.minimize(self.loss,
                global_step=tf.contrib.framework.get_global_step())
        with tf.name_scope('target_network_update'):
            q_network_params = [t for t in tf.trainable_variables() if t.name.startswith(
                'q_network')]
            q_network_params = sorted(q_network_params, key=lambda v: v.name)

            target_network_params = [t for t in tf.trainable_variables() if t.name.startswith(
                'target_q_network')]
            target_q_network_params = sorted(target_q_network_params, key=lambda v: v.name)

            self.target_update_ops = []
            for e1_v, e2_v in zip(q_network_params, target_q_network_params):
                op = e2_v.assign(e1_v)
                self.target_update_ops.append(op)

    def sample(self, state):
        self._current_time_step += 1
        q_values = self.sess.run(self.q_values, {self.state: state})

        epsilon = self._epsilons[min(self._current_time_step, self._epsilon_decay_steps)]

        e = random.random()
        if e < epsilon:
            return random.randint(0, self._num_actions - 1)
        else:
            return np.argmax(q_values)

    def best_action(self, state):
        q_values = self.sess.run(self.q_values, {self.state: state})
        return np.argmax(q_values)

    def store(self, state, action, reward, next_state, terminal):
        self._replay_buffer.add(state, action, reward, next_state, terminal)

    def update(self):
        states, actions, rewards, next_states, terminals = self._replay_buffer.sample()

        next_states_q_values = self.sess.run(self._target_q_values, {self._state: next_states})
        next_states_max_q_values = np.max(next_states_q_values, axis=1)

        if len(states) > 0:
            td_targets = rewards + (1 - terminals) * self.DISCOUNT * next_states_max_q_values

            feed_dict = {self._state: states,
                         self._picked_actions: actions,
                         self._td_targets: td_targets}

            _ = sess.run(self.train_op, feed_dict=feed_dict)

        # Update the target q-network.
        if not self._current_time_step % self.TARGET_UPDATE:
            sess.run(self.target_update_ops)
