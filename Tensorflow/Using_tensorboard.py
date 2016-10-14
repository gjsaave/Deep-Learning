import tensorflow as tf
import numpy as np

x = tf.constant([35, 40, 45], name='x')
y = tf.Variable(x + 5, name='y')

with tf.Session() as session:
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("/tmp/basic", session.graph)
    model = tf.initialize_all_variables()
    session.run(model)
    print(session.run(y))