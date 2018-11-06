from root.common.hparams import params as pm
from tensorflow.nn.rnn_cell import RNNCell, GRUCell, MultiRNNCell
from tensorflow.python.ops import variable_scope
from collections import namedtuple


class wavernn_cell(RNNCell):

    def __init__(self,
                 coarse_units=pm.seq_cells,
                 fine_units=pm.seq_cells,
                 train_via_gt=True):

        with variable_scope("coarse"):
            self.coarse_cell = MultiRNNCell(
                [GRUCell(cell_units) for cell_units in coarse_units])

        with variable_scope("fine"):
            self.fine_cell = MultiRNNCell(
                [GRUCell(cell_units) for cell_units in fine_units])

        assert coarse_units[-1] == fine_units[-1]

        self.is_gt = train_via_ground_truth
        self.container = namedtuple('wavernn', ['coarse', 'fine'])

    @property
    def output_size(self):
        return self.container(coarse=self.coarse_cell.output_size,
                              fine=self.fine_cell.output_size)

    @property
    def state_size(self):
        return self.container(coarse=self.coarse_cell.state_size,
                              fine=self.fine_cell.state_size)

    def zero_state(self, batch_size, dtype):
        self.current_input = self.container(coarse=tf.zeros(batch_size),
                                            fine=tf.zeros(batch_size))
        return self.container(
            coarse=self.coarse_cell.zero_state(batch_size, dtype),
            fine=self.fine_cell.zero_state(batch_size, dtype))

    def __call__(self, input_data, state, scope):

        with tf.name_scope(scope):
            coarse_input = tf.stack([self.current_input[0],
                                     self.current_input[1],
                                     input_data[0]], axis=-1,
                                    name="input/coarse")

            if self.is_gt:
                fine_input = tf.stack([self.current_input[0],
                                       self.current_input[1],
                                       input_data[1]], axis=-1,
                                      name="input/fine/gt")

                self.current_input = self.container(
                    coarse=self.current_input[1],
                    fine=self.current_input[2])

            coarse_output, coarse_state = self.coarse_cell(
                coarse_input, state.coarse, scope="{}/coarse".format(scope))

            if not self.is_gt:
                coarse_scaled = tf.reduce_max(tf.nn.softmax(coarse_output),
                                              axis=1, name="reduction/coarse")/256

                fine_input = tf.stack([self.current_input[0],
                                       self.current_input[1],
                                       coarse_scaled], axis=-1,
                                      name="input/fine/out")

                fine_output, fine_state = self.fine_cell(
                    fine_input, state.fine, scope"{}/fine".format(scope))

                fine_scaled = tf.reduce_max(tf.nn.softmax(fine_output),
                                            axis=1, name="reduction/fine")/256

                self.current_input = self.container(
                    coarse=coarse_scaled, fine=fine_scaled)

        return (self.container(coarse=coarse_output, fine=fine_output),
                self.container(coarse=coarse_state, fine=fine_state))
