import os
from glob import glob
import tensorflow as tf
import numpy as np
from root.common.hparams import params


class input_feeder:
    """
        Batch only takes one audio sample per training loop
        due to rnn space constraints.
    """

    def __init__(self):
        self.__input__()

    def __py_controller__(self, file):

        each_sample = np.load(file.decode("utf-8"))
        mel = each_sample['mel']
        inputs = each_sample['input']
        ground_truth = each_sample['ground_truth']
        return (mel.astype(np.float32),
                inputs.astype(np.float32),
                ground_truth.astype(np.int32))

    def __input__(self):

        try:
            training_files = glob("{}/*.npz".format(params.train_data_dir))
            dataset = tf.data.Dataset.from_tensor_slices(
                list(training_files))
            # shuffled = dataset.shuffle(100)
            np_dataset = shuffled.map(lambda file_name: tf.py_func(
                func=self.__import_batchless__, inp=[file_name],
                Tout=[tf.float32,  tf.float32, tf.int32]))
            self.iterator = np_dataset.make_initializable_iterator()
        except Exception as e:
            print("Meta data not found, check hparams or raise issue on git")
            exit()
