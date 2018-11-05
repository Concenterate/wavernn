import tensorflow as tf
from wavernn.hparams import params as pm
from wavernn.feeder import input_feeder
from wavernn.model.models import custom_rnn
from datetime import datetime


class wrapper:

    def __inputs__(self):

        iterator = input_feeder().training_iterator
        self.initializer = iterator.initializer
        self.mel, self.linear = iterator.get_next()
        self.linear.set_shape([2, None])
        self.mel.set_shape([None, pm.num_mels])

    def __architecture__(self):

        with tf.variable_scope("mel_embedding"):

            self.coarse_local = tf.layers.dense(
                inputs=self.mel, units=256)

        if pm.model_type is 'wavernn':

            with tf.name_scope("wavernn"):
                self.seq_cell = custom_rnn(gt=True)
                self.output = self.seq_cell(
                    self.coarse_local, self.linear[0], self.linear[1])

    def __init__(self):

        self.__inputs__()
        self.__architecture__()
