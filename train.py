
import tensorflow as tf
from wavernn.model.wavernn import wrapper

c = wrapper()

session = tf.Session()
var = tf.global_variables_initializer()
session.run(var)
session.run(c.initializer)
print(session.run(c.output))
