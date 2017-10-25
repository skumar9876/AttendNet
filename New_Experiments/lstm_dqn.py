from dqn import DqnAgent
import tensorflow as tf

class LstmDqnAgent(DqnAgent):

    def __init__(self, sequence_length=0, *args, **kwargs):
        self._sequence_length = sequence_length
        super(LstmDqnAgent, self).__init__(*args, **kwargs)

    def _q_network(self, state):
        print "Meta-Controller!"
        print state.get_shape()

        embeddings = tf.get_variable('embeddings',
            [self._num_actions + 1, 128])

        embedded_ids = tf.gather(embeddings, state)

        print embedded_ids.get_shape()

        lstm = tf.contrib.rnn.BasicLSTMCell(128)

        cell_state, hidden_state = tf.nn.dynamic_rnn(
            cell=lstm, inputs=embedded_ids, dtype=tf.float32)

        print "Number of actions:"
        print self._num_actions

        print hidden_state[1]
        print cell_state

        q_values = tf.contrib.layers.fully_connected(
            hidden_state[1], self._num_actions, activation_fn=None)

        return q_values

    def _construct_graph(self):
        shape=[None]
        for dim in self._state_dims:
            shape.append(dim)
        self._state = tf.placeholder(shape=shape, dtype=tf.int32)

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


