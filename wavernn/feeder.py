import os
from glob import glob
import tensorflow as tf
import numpy as np
from wavernn.hparams import params


class input_feeder:

    def __init__(self):

        self.__input__()

    def __import_data__(self, file):

        single_audio_sample = np.load(file.decode("utf-8"))
        mel_data = single_audio_sample['mel']
        linear_data = single_audio_sample['linear']

        return mel_data.astype(np.float32), linear_data.astype(np.float32)

    def __input__(self):

        try:

            meta_data_file = open(os.path.join(params.train_data_dir,
                                               params.meta_data_file_name), 'r')
            training_files = map(lambda url: url.split("|")[0],
                                 meta_data_file.readlines())
            dataset = tf.data.Dataset.from_tensor_slices(list(training_files))
            np_dataset = dataset.shuffle(100)
            np_dataset = np_dataset.map(lambda file_name: tf.py_func(
                func=self.__import_data__, inp=[file_name],
                Tout=[tf.float32, tf.float32]))
            self.training_iterator = np_dataset.make_initializable_iterator()
        except Exception as e:
            print("Meta data not found, check hparams or raise issue on git")
            exit()


# if __name__ == "__main__":
#     f = input_feeder()
#     sess = tf.Session()
#     sess.run(f.training_iterator.initializer)
#     print(sess.run(f.training_iterator.get_next()))
