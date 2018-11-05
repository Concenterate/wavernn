from tensorflow.nn.rnn_cell import RNNCell, GRUCell, MultiRNNCell
import tensorflow as tf
from common.hparams import params as pm


class WNNCell(RNNCell):

    def __init__(self, train_via_ground_truth):

        with tf.variable_scope("coarse"):
            self.coarse_cell = MultiRNNCell(
                [GRUCell(num_units) for num_units in pm.num_cell])
        with tf.variable_scope("fine"):
            self.fine_cell = MultiRNNCell(
                [GRUCell(num_units) for num_units in [256, 256]])

        self.is_gt = train_via_ground_truth

    def __call__(self, input_data, state):

        mel_data, coarse_t, fine_t = input_data

        coarse_input = tf.stack([self.p_e[0], self.p_e[1], mel_data],
                                axis=-1, name="coarse_input")

        with tf.variable_scope("coarse", reuse=tf.AUTO_REUSE):
            coarse_output, coarse_state = self.coarse_cell(
                coarse_input, state[0])

        if self.is_gt:
            fine_input = tf.stack([self.p_e[0], self.p_e[1], coarse_t],
                                  axis=-1, name="fine_input_gt")
        else:
            with tf.name_scope("coarse_reduction"):
                coarse_scaled = tf.reduce_max(tf.nn.softmax(coarse_output), axis=1,
                                              name="reduced")/256

                fine_input = tf.stack([self.p_e[0], self.p_e[1], coarse_scaled],
                                      axis=-1, name="fine_input_out")

        with tf.variable_scope("fine", reuse=tf.AUTO_REUSE):
            fine_output, fine_state = self.fine_cell(fine_input, state[1])

            with tf.name_scope("fine_reduction"):
                fine_scaled = tf.reduce_max(tf.nn.softmax(fine_output),
                                            axis=1, name="reduced")/256

        if self.is_gt:
            self.previous_element = (coarse_t, fine_t)
        else:
            self.previous_element = (coarse_scaled, fine_scaled)

        # output = tf.concat([coarse_output, fine_output],
        #                    axis=-1, name="rnn_outp")

        return (coarse_output, fine_output), (coarse_state, fine_state)

    def zero_state(self, batch_size, dtype):
        self.p_e = (tf.zeros(batch_size), tf.zeros(batch_size))
        return (self.coarse_cell.zero_state(batch_size, dtype),
                self.fine_cell.zero_state(batch_size, dtype))

    @property
    def output_size(self):
        return (256, 256)

    @property
    def state_size(self):
        return (self.coarse_cell.state_size, self.fine_cell.state_size)


class custom_rnn:

    def __init__(self, gt):

        self.rnn_cell = sequential_cell(gt)
        self._gt = gt

    def reshape_and_pad(self, arr, pad_length):
        padded_seq = tf.pad(arr, [[0, pad_length]])
        return tf.reshape(padded_seq, [-1, pm.seq_length])

    def sequencer(self, time, cell_output, cell_state, loop_state):

        print(time, cell_output, cell_state, loop_state)
        if cell_output is None:
            next_cell_state = self.rnn_cell.zero_state(
                self._batch_size, tf.float32)
        else:
            next_cell_state = cell_state

        elements_finished = (time >= pm.seq_length)

        if self._gt:
            next_input = tf.cond(
                elements_finished,
                lambda: (
                    self.batch_embedded_mel[time], self.coarse_frames[time], self.fine_frames[time]),
                lambda: (self.batch_embedded_mel[time], self.coarse_frames[time], self.fine_frames[time]))
        else:
            next_input = tf.cond(
                elements_finished,
                lambda: (
                    self.batch_embedded_mel[time], self.coarse_frames[time], self.fine_frames[time]),
                lambda: (self.batch_embedded_mel[time]))
        next_loop_state = None
        return (elements_finished, next_input, next_cell_state,
                cell_output, next_loop_state)

    def __call__(self, embedded_mel, coarse_frames, fine_frames):
        print(embedded_mel)

        flattended_mels = tf.reshape(embedded_mel, shape=[-1])
        pad_length = tf.mod(tf.shape(flattended_mels)[0], pm.seq_length)
        self.batch_embedded_mel = self.reshape_and_pad(
            flattended_mels, pad_length)
        print(self.batch_embedded_mel)
        if self._gt:
            self.coarse_frames = self.reshape_and_pad(
                coarse_frames, pad_length)
            self.fine_frames = self.reshape_and_pad(fine_frames, pad_length)

        self._batch_size = tf.shape(self.batch_embedded_mel)[0]

        return tf.nn.raw_rnn(cell=self.rnn_cell, loop_fn=self.sequencer)
