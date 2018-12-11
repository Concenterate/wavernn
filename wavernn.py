import tensorflow as tf
from root.common.feeder import input_feeder
from root.common.hparams import params
from root.model.wavernn_cell import wavernn_cell
from root.model.seq_encoder import wavernn_encoder
from tqdm import tqdm


class wavernn896:

    def __inputs__(self):

        self.data_iterator = input_feeder().iterator

        self.mel_data, self.linear_input, self.linear_ground_truth = self.data_iterator.get_next()

        self.mel_data.set_shape([None, params.num_mels])
        self.linear_input.set_shape([2, None])
        self.linear_ground_truth.set_shape([2, None])

    def __prenet__(self, network="dense"):

        with tf.name_scope("prenet"):

            if network == "dense":

                embedding = tf.layers.dense(inputs=self.mel_data,
                                            units=params.hop_size,
                                            name="embed")

            elif network == "cnn":

                embedding = tf.layers.dense(inputs=self.mel_data,
                                            units=params.hop_size,
                                            name="embed")

            elif network == "rnn":

                embedding = tf.layers.dense(inputs=self.mel_data,
                                            units=params.hop_size,
                                            name="embed")

            return embedding

    def __seqnet__(self):

        with tf.name_scope("seq_net"):

            seq_cell = wavernn_cell()
            mel_embedding = self.__prenet__()
            encoder = wavernn_encoder(mel_embedding,
                                      self.linear_input[0],
                                      self.linear_input[1])
            output = encoder()
        return output

    def __loss__(self, seq_output):

        gt0 = self.linear_ground_truth[0]
        gt1 = self.linear_ground_truth[1]

        coarse_labels = tf.one_hot(gt0, depth=256)
        fine_labels = tf.one_hot(gt1, depth=256)

        c_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=seq_output[0],
                                                            labels=coarse_labels)
        f_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=seq_output[1],
                                                            labels=fine_labels)

        c_sum = tf.reduce_sum(c_loss)
        f_sum = tf.reduce_sum(f_loss)
        return [c_sum, f_sum]

    def __init__(self):

        self.__inputs__()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        self.session.run(self.data_iterator.initializer)
        self.seq_output = self.__seqnet__()
        self.loss = self.__loss__(self.seq_output)
        self.opt = tf.train.AdadeltaOptimizer(0.9).minimize(
            self.loss[0]+self.loss[1])
        global_var = tf.global_variables_initializer()
        self.session.run(global_var)
        # while(True):
        #     try:
        #         self.session.run(opt)
        #         seq.update()
        #         seq.set_postfix(
        #             {"loss": self.session.run(tf.reduce_mean(loss))})
        #     except tf.errors.OutOfRangeError:
        #         session.run(self.data_iterator.initializer)
        #         continue

        # print(session.run(self.linear_ground_truth[0][0]))
