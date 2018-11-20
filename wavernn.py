import tensorflow as tf
from root.common.feeder import input_feeder
from root.common.hparams import params
from root.model.wavernn_cell import wavernn_cell
from root.model.wave_encoder import wavernn_encoder
from tqdm import tqdm


class wavernn896:

    def __inputs__(self):

        self.data_iterator = input_feeder().iterator

        self.mel_data, self.linear_input, self.linear_ground_truth = self.data_iterator.get_next()
        self.mel_data.set_shape([None, params.num_mels])
        self.linear_input.set_shape([2, None])
        self.linear_ground_truth.set_shape([2, None])

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
                                      self.linear_input[0],
                                      self.linear_input[1])
            output, final_state = encoder()
        return output

    def __loss__(self, seq_output):

        coarse_labels = tf.one_hot(self.linear_ground_truth[0], depth=256)
        c_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=seq_output.coarse,
                                                            labels=coarse_labels)

        fine_labels = tf.one_hot(self.linear_ground_truth[1], depth=256)
        f_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=seq_output.fine,
                                                            labels=fine_labels)

        return c_loss+f_loss

    def __init__(self):

        self.__inputs__()
        session = tf.Session()
        session.run(self.data_iterator.initializer)
        seq_output = self.__seqnet__()
        loss = self.__loss__(seq_output)
        opt = tf.train.AdadeltaOptimizer().minimize(loss)
        global_var = tf.global_variables_initializer()
        session.run(global_var)
        seq = tqdm()
        bp = 100
        print(params.seq_cells)
        while(True):
            try:
                session.run(opt)
                seq.update()
                seq.set_postfix({"loss": session.run(tf.reduce_mean(loss))})
            except tf.errors.OutOfRangeError:
                session.run(self.data_iterator.initializer)
                continue

        print(session.run(self.linear_ground_truth[0][0]))
