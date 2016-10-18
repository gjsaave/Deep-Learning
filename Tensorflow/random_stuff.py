import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


# x and y are placeholders for our training data
x = tf.placeholder("float")
y = tf.placeholder("float")
# w is the variable storing our values. It is initialised with starting "guesses"
# w[0] is the "a" in our equation, w[1] is the "b"
w = tf.Variable([1.0, 2.0], name="w")
# a = tf.Variable(1.0, name="a")
# b = tf.Variable(1.0, name="b")

# Our model of y = a*x + b
# y_model = tf.mul(w[0], x) + w[1]
y_model = tf.scalar_mul(w[0], x) + w[1]


# Our error is defined as the square of the differences
error = tf.square(y - y_model)
# The Gradient Descent Optimizer does the heavy lifting
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(error)

# Normal TensorFlow - initialize values, create a session and run the model
model = tf.initialize_all_variables()

# errors = []
# with tf.Session() as session:
#     session.run(model)
#     for i in range(20):
#         x_train = tf.random_normal((50,), mean=5, stddev=2.0)
#         # x_train = np.random.rand(50)
#         y_train = x_train * 2 + 6
#         x_value, y_value = session.run([x_train, y_train])
#         # print x_value
#         # print y_value
#         _, error_value = session.run([train_op, error], feed_dict={x: x_value, y: y_value})
#         print (session.run(train_op, feed_dict={x: x_value, y: y_value}))
#         errors.append(error_value)
#     w_value = session.run(w)
#
#     print("Predicted model: {a:.3f}x + {b:.3f}".format(a=w_value[0], b=w_value[1]))
#
# plt.plot([np.mean(errors[i-50:i]) for i in range(len(errors))])
# plt.show()
# plt.savefig("errors.png")

with tf.Session() as session:
    session.run(model)
    for i in range(20):
        x_value = np.random.rand(50)
        y_value = x_value * 2 + 6
        print x_value
        print y_value
        session.run(train_op, feed_dict={x: x_value, y: y_value})
        print (session.run(train_op, feed_dict={x: x_value, y: y_value}))


    w_value = session.run(w)
    print("Predicted model: {a:.3f}x + {b:.3f}".format(a=w_value[0], b=w_value[1]))