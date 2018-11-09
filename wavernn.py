import tensorflow as tf
from root.common.feeder import input_feeder
from root.common.hparams import params
from root.model.wavernn_cell import wavernn_cell
from root.model.wave_encoder import wavernn_encoder


class wavernn896:

    def __inputs__(self):

        self.data_iterator = input_feeder().iterator

        self.mel_data, self.linear_input, self.linear_ground_truth = self.data_iterator.get_next()
        self.mel_data.set_shape([None, params.num_mels])
        self.linear_input.set_shape([None, 2])
        self.linear_ground_truth.set_shape([None, 2])

    def __prenet__(self):

        with tf.name_scope("prenet"):

            embedding = tf.layers.dense(inputs=self.mel_data,
                                        units=params.hop_size,
                                        name="embed")

            return tf.layers.dropout(inputs=embedding,
                                     rate=params.prenet_dropout)

    def __seqnet__(self):

        with tf.name_scope("prenet"):

            seq_cell = wavernn_cell()
            mel_embedding = self.__prenet__()
            encoder = wavernn_encoder(mel_embedding,
                                      self.linear_input[:, 0],
                                      self.linear_input[:, 1])
            output, final_state = encoder()
        print(final_state)
        return output

    def __init__(self):

        self.__inputs__()
        session = tf.Session()
        session.run(self.data_iterator.initializer)
        # session.run(self.mel_data)
        self.__seqnet__()
