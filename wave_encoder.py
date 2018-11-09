from root.common.hparams import params
from tensorflow.python.ops import array_ops, control_flow_ops, gen_array_ops, gen_math_ops
from tensorflow.nn import raw_rnn
from root.model.wavernn_cell import wavernn_cell


class wavernn_encoder:

    """
        It works as an alternative to dynamic_rnn because we can not
        give input to wavernn_cell using dynamic_rnn.

        This encoder only supports wavernn module.

    """

    def __init__(self, embedded_mel, coarse_frames=None, fine_frames=None):

        flattended_mels = gen_array_ops.reshape(embedded_mel, shape=[-1])
        pad_length = gen_math_ops.mod(array_ops.shape(
            flattended_mels)[0], params.seq_length)

        self.batch_embedded_mel = self.reshape_and_pad(
            flattended_mels, pad_length)
        self._batch_size = array_ops.shape(self.batch_embedded_mel)[0]

        self.is_train = False
        if coarse_frames is not None:
            self.coarse_frames = self.reshape_and_pad(
                coarse_frames, pad_length)
            self.fine_frames = self.reshape_and_pad(fine_frames, pad_length)
            self.is_train = True
        self.rnn_cell = wavernn_cell(training=self.is_train)

    def reshape_and_pad(self, arr, pad_length):
        padded_seq = array_ops.pad(arr, [[0, pad_length]])
        return gen_array_ops.reshape(padded_seq, [-1, params.seq_length])

    def sequencer(self, time, cell_output, cell_state, loop_state):

        if cell_output is None:
            next_cell_state = self.rnn_cell.zero_state(
                self._batch_size, self.batch_embedded_mel.dtype)
        else:
            next_cell_state = cell_state

        elements_finished = (time >= params.seq_length)

        if self.is_train:
            next_input = control_flow_ops.cond(
                elements_finished,
                lambda: (array_ops.zeros([1, params.seq_length]),
                         array_ops.zeros([1, params.seq_length]),
                         array_ops.zeros([1, params.seq_length])),
                lambda: (self.batch_embedded_mel[time],
                         self.coarse_frames[time],
                         self.fine_frames[time]))
        else:
            next_input = control_flow_ops.cond(
                elements_finished,
                lambda: (array_ops.zeros([1, params.seq_length]),),
                lambda: (self.batch_embedded_mel[time]),)

        return (elements_finished, next_input, next_cell_state,
                cell_output, loop_state)

    def __call__(self):

        output, state, _ = raw_rnn(
            cell=self.rnn_cell, loop_fn=self.sequencer, scope="wavernn_encoder")

        output = self.rnn_cell.container(coarse=output.coarse.stack(name="output"),
                                         fine=output.fine.stack(name="output"))
        return (output, state)
