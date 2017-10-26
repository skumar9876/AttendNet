from dqn import DqnAgent
import tensorflow as tf

class ControllerDqnAgent(DqnAgent):


    def __init__(self, subgoal_dims=[], *args, **kwargs):
        self._subgoal_dims = subgoal_dims
        super(ControllerDqnAgent, self).__init__(*args, **kwargs)

    def _q_network(self, state, subgoal):
        conv1 = tf.layers.conv2d(inputs=state, filters=8, kernel_size=[5, 5], padding='valid', activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=1)
        conv2 = tf.layers.conv2d(inputs=pool1, filters=16, kernel_size=[5, 5], padding='valid', activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=1)
        conv3 = tf.layers.conv2d(inputs=pool2, filters=32, kernel_size=[3, 3], padding='valid', activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=1)
        pool3_flat = tf.contrib.layers.flatten(pool3)

        subgoal_layer1 = tf.contrib.layers.fully_connected(subgoal, 64, activation_fn=tf.nn.relu)

        concatenated = tf.concat([pool3_flat, subgoal_layer1], axis=1)

        dense1 = tf.contrib.layers.fully_connected(concatenated, 64, activation_fn=tf.nn.relu)
        q_values = tf.contrib.layers.fully_connected(dense1, self._num_actions, activation_fn=None)

        return q_values

    def _construct_graph(self):
        state_shape=[None]
        subgoal_shape=[None]
        for dim in self._state_dims:
            state_shape.append(dim)
        for dim in self._subgoal_dims:
            subgoal_shape.append(dim)

        self._state = tf.placeholder(shape=state_shape, dtype=tf.float32)
        self._subgoal = tf.placeholder(shape=subgoal_shape, dtype=tf.float32)

        with tf.variable_scope('q_network'):
            self._q_values = self._q_network(self._controller_state, self._subgoal)
        with tf.variable_scope('target_q_network'):
            self._target_q_values = self._q_network(self._controller_state, self._subgoal)
        with tf.variable_scope('q_network_update'):
            self._picked_actions = tf.placeholder(shape=[None, 2], dtype=tf.int32)
            self._td_targets = tf.placeholder(shape=[None], dtype=tf.float32)

            self._q_values_pred = tf.gather_nd(self._q_values, self._picked_actions)
            self._losses = tf.squared_difference(self._q_values_pred, self._td_targets)
            self._loss = tf.reduce_mean(self._losses)

            self.optimizer = tf.train.RMSPropOptimizer(self._learning_rate)
            # self.optimizer = tf.train.RMSPropOptimizer(self._learning_rate, 0.99, 0.0, 1e-6)
            # self.optimizer = tf.train.AdamOptimizer(0.0001)
            # self.optimizer = tf.train.GradientDescentOptimizer(0.1)
            self.train_op = self.optimizer.minimize(self._loss,
                global_step=tf.contrib.framework.get_global_step())
        with tf.name_scope('target_network_update'):
            q_network_params = [t for t in tf.trainable_variables() if t.name.startswith(
                'q_network')]
            q_network_params = sorted(q_network_params, key=lambda v: v.name)

            target_q_network_params = [t for t in tf.trainable_variables() if t.name.startswith(
                'target_q_network')]
            target_q_network_params = sorted(target_q_network_params, key=lambda v: v.name)

            self.target_update_ops = []
            for e1_v, e2_v in zip(q_network_params, target_q_network_params):
                op = e2_v.assign(e1_v)
                self.target_update_ops.append(op)
