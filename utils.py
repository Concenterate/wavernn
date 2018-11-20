from tensorflow.python.ops import array_ops, gen_array_ops
from root.common.hparams import params


def reshape_and_pad(arr, pad_length):
    padded_seq = array_ops.pad(arr, [[0, pad_length]])
    return gen_array_ops.reshape(padded_seq, [-1, params.seq_length])
