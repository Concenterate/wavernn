from root.common.hparams import params
# from root.common.utils import reshape_and_pad
from tensorflow.python.ops import array_ops, control_flow_ops, gen_array_ops, gen_math_ops, math_ops, logging_ops
from tensorflow.nn import raw_rnn
from root.model.wavernn_cell import wavernn_cell
from tensorflow.python.framework.dtypes import int32 as tf_int32
import tensorflow as tf


class wavernn_encoder:

    """
        It works as an alternative to dynamic_rnn because we can not
        give input to wavernn_cell using dynamic_rnn.

        This encoder only supports wavernn module.

    """

    def __init__(self, local_cond, coarse_samples=None, fine_samples=None):

        with tf.name_scope("features_restruction"):
            num_of_samples = array_ops.shape(coarse_samples)[0]

            self.pad_length = params.seq_length-gen_math_ops.mod(
                num_of_samples,
                params.seq_length)

            self.local_cond_feat = array_ops.pad(
                tf.reshape(local_cond, [-1]), [[0, self.pad_length]])

            self._batch_size = tf.cast(math_ops.floordiv(
                num_of_samples, params.seq_length)+1, tf.int32)

            self.is_train = False
            if coarse_samples is not None:
                self.coarse_frames = array_ops.pad(
                    coarse_samples,  [[0, self.pad_length]])
                self.fine_frames = array_ops.pad(
                    fine_samples,  [[0, self.pad_length]])
                self.is_train = True

        self.rnn_cell = wavernn_cell(training=self.is_train)

    def sequencer(self, time, cell_output, cell_state, loop_state):

        if cell_output is None:
            next_cell_state = self.rnn_cell.zero_state(
                self._batch_size, self.local_cond_feat.dtype)
        else:
            next_cell_state = cell_state

        elements_finished = (time >= params.seq_length)
        batch_index = self._batch_size * time

        if self.is_train:
            next_input = control_flow_ops.cond(
                elements_finished,
                lambda: (array_ops.zeros([self._batch_size]),
                         array_ops.zeros([self._batch_size]),
                         array_ops.zeros([self._batch_size])),
                lambda: (self.local_cond_feat[batch_index:batch_index+self._batch_size],
                         self.coarse_frames[batch_index:batch_index +
                                            self._batch_size],
                         self.fine_frames[batch_index:batch_index+self._batch_size]))
        else:
            next_input = control_flow_ops.cond(
                elements_finished,
                lambda: (array_ops.zeros([self._batch_size]),),
                lambda: (self.batch_embedded_mel[time:time+self._batch_size]),)

        return (elements_finished, next_input, next_cell_state,
                cell_output, loop_state)

    def __call__(self):

        output, state, _ = raw_rnn(
            cell=self.rnn_cell, loop_fn=self.sequencer, scope="wavernn_encoder")

        coarse = output.coarse.concat(name="output")[:-self.pad_length]
        fine = output.fine.concat(name="output")[:-self.pad_length]

        output = (coarse, fine)
        return output
