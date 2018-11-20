from root.common.hparams import params
from root.common.utils import reshape_and_pad
from tensorflow.python.ops import array_ops, control_flow_ops, gen_array_ops, gen_math_ops, math_ops, logging_ops
from tensorflow.nn import raw_rnn
from root.model.wavernn_cell import wavernn_cell
from tensorflow.python.framework.dtypes import int32 as tf_int32


class wavernn_encoder:

    """
        It works as an alternative to dynamic_rnn because we can not
        give input to wavernn_cell using dynamic_rnn.

        This encoder only supports wavernn module.

    """

    def __init__(self, embedded_mel, coarse_frames=None, fine_frames=None):

        flattended_mels = gen_array_ops.reshape(embedded_mel, shape=[-1])

        self.pad_length = math_ops.subtract(params.seq_length,
                                            gen_math_ops.mod(
                                                array_ops.shape(
                                                    flattended_mels)[0],
                                                params.seq_length))

        self.batch_embedded_mel = array_ops.pad(
            flattended_mels, [[0, self.pad_length]])
        self._batch_size = math_ops.cast(array_ops.shape(self.batch_embedded_mel)[
            0]/params.seq_length, tf_int32)

        self.is_train = False
        if coarse_frames is not None:
            self.coarse_frames = array_ops.pad(
                coarse_frames, [[0, self.pad_length]])
            self.fine_frames = array_ops.pad(
                fine_frames, [[0, self.pad_length]])
            self.is_train = True
        self.rnn_cell = wavernn_cell(training=self.is_train)

    def sequencer(self, time, cell_output, cell_state, loop_state):
        if cell_output is None:
            next_cell_state = self.rnn_cell.zero_state(
                self._batch_size, self.batch_embedded_mel.dtype)
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
                lambda: (self.batch_embedded_mel[batch_index:batch_index+self._batch_size],
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
            cell=self.rnn_cell, loop_fn=self.sequencer, scope="wavernn_encoder", swap_memory=True)

        coarse = output.coarse.concat(name="output")[:-self.pad_length]
        fine = output.fine.concat(name="output")[:-self.pad_length]

        output = self.rnn_cell.container(coarse=coarse,
                                         fine=fine)
        return (output, state)
