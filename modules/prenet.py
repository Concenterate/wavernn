

class prenet:

    def __init__(self, optype, scope):

        self._optype = optype
        self._scope = scope

    def __call__(self, inputs):

        with tf.variable_scope(self._scope):

                # if self._optype == "dense":

            output = tf.layers.dense(
                inputs=inputs, units=params.hop_size, name="deep")

# elif network == "cnn":

#      embedding = tf.layers.dense(inputs=self.mel_data,
#                                  units=params.hop_size,
#                                  name="convolutional")

#  elif network == "rnn":

#      embedding = tf.layers.dense(inputs=self.mel_data,
#                                  units=params.hop_size,
#                                  name="recurrent")


return output
