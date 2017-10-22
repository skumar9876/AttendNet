from dqn import DqnAgent
import tensorflow as tf

class LstmDqnAgent(DqnAgent):

    def __init__(self, sequence_length=0, *args, **kwargs):
        self._sequence_length = sequence_length
        super(LstmDqnAgent, self).__init__(*args, **kwargs)

    def _q_network(self, state):

        print state.get_shape()

        lstm = tf.contrib.rnn.BasicLSTMCell(128)

        cell_state, hidden_state = tf.contrib.rnn.static_rnn(
            cell=lstm, inputs=state,
            sequence_length=self._sequence_length)

        q_values = tf.contrib.fully_connected(hidden_state, self._num_actions, activation_fn=None)

        return q_values


