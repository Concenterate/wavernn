from root.common.hparams import params
from tensorflow.nn.rnn_cell import RNNCell, GRUCell, MultiRNNCell
from tensorflow.python.ops.variable_scope import variable_scope
from tensorflow.python.framework.ops import name_scope
from tensorflow.python.ops import array_ops, math_ops, nn_ops
from tensorflow.python.layers import core
from collections import namedtuple
import tensorflow as tf


class wavernn_cell(RNNCell):

    def __init__(self,
                 coarse_units=params.seq_cells,
                 fine_units=params.seq_cells,
                 training=True):
        """

            coarse_units: sequential cell details for coarse sub cell
            fine_units: sequential cell details for fine sub cell
            training: whether the loop is training or predicting
                It is useful because while training, cell expects a tuple of
                size 3 as input to condition the network using ground truth,
                while during prediction it expects tuple of size 1 as local
                condition parameter.
        """

        with variable_scope("coarse"):
            self.coarse_cell = MultiRNNCell(
                [GRUCell(cell_units) for cell_units in coarse_units])

        with variable_scope("fine"):
            self.fine_cell = MultiRNNCell(
                [GRUCell(cell_units) for cell_units in fine_units])

        assert coarse_units[-1] == fine_units[-1]

        self.is_train = training
        self.container = namedtuple('wavernn', ['coarse', 'fine'])

    @property
    def output_size(self):
        return self.container(coarse=params.rnn_resolution,
                              fine=params.rnn_resolution)

    @property
    def state_size(self):
        return self.container(coarse=self.coarse_cell.state_size,
                              fine=self.fine_cell.state_size)

    def zero_state(self, batch_size, dtype):
        self.current_input = self.container(coarse=array_ops.zeros(batch_size),
                                            fine=array_ops.zeros(batch_size))
        return self.container(
            coarse=self.coarse_cell.zero_state(batch_size, dtype),
            fine=self.fine_cell.zero_state(batch_size, dtype))

    def post_net_layer(self, X):

        with variable_scope("post_net"):
            return core.dense(inputs=X, units=params.rnn_resolution)

    def __call__(self, input_data, state, scope="wavernn_cell"):

        with name_scope(scope):
            coarse_input = array_ops.stack([self.current_input[0],
                                            self.current_input[1],
                                            input_data[0]], axis=-1,
                                           name="input/coarse")

            coarse_output, coarse_state = self.coarse_cell(
                coarse_input, state.coarse,
                scope="{}/coarse".format(scope))

            coarse_output = core.dense(inputs=coarse_output,
                                       units=params.rnn_resolution)

            if self.is_train:
                fine_input = array_ops.stack([self.current_input[0],
                                              self.current_input[1],
                                              input_data[1]], axis=-1,
                                             name="input/fine/gt")

                self.current_input = self.container(
                    coarse=input_data[1],
                    fine=input_data[2])
            else:
                coarse_scaled = math_ops.divide(
                    math_ops.reduce_max(
                        nn_ops.softmax(coarse_output), axis=1),
                    y=256, name="reduction/coarse")

                fine_input = array_ops.stack([self.current_input[0],
                                              self.current_input[1],
                                              coarse_scaled], axis=-1,
                                             name="input/fine/out")

            fine_output, fine_state = self.fine_cell(
                fine_input, state.fine, scope="{}/fine".format(scope))

            fine_output = core.dense(inputs=fine_output,
                                     units=params.rnn_resolution)

            if not self.is_train:

                fine_scaled = math_ops.divide(
                    math_ops.reduce_max(
                        nn_ops.softmax(fine_output),
                        axis=1),
                    y=256, name="reduction/fine")

                self.current_input = self.container(
                    coarse=coarse_scaled, fine=fine_scaled)
        return (self.container(coarse=coarse_output, fine=fine_output),
                self.container(coarse=coarse_state, fine=fine_state))
