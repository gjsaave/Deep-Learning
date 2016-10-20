import tensorflow as tf

# [1, 2, 3, 4 ,...]
x = tf.range(1, 10, name="x")

# A queue that outputs 0,1,2,3,...
range_q = tf.train.range_input_producer(limit=5, shuffle=False)
slice_end = range_q.dequeue()

# Slice x to variable length, i.e. [0], [0, 1], [0, 1, 2], ....
y = tf.slice(x, [0], [slice_end], name="y")

# Batch the variable length tensor with dynamic padding
batched_data = tf.train.batch(
    tensors=[y],
    batch_size=5,
    dynamic_pad=True,
    name="y_batch"
)

# Run the graph
# tf.contrib.learn takes care of starting the queues for us
res = tf.contrib.learn.run_n({"z": batched_data}, n=1, feed_dict=None)

# Print the result
print("Batch shape: {}".format(res[0]["z"].shape))
print(res[0]["z"])


##############

# Creating a new queue
padding_q = tf.PaddingFIFOQueue(
    capacity=10,
    dtypes=tf.int32,
    shapes=[[None]])

# Enqueue the examples
enqueue_op = padding_q.enqueue([y])

# Add the queue runner to the graph
qr = tf.train.QueueRunner(padding_q, [enqueue_op])
tf.train.add_queue_runner(qr)

# Dequeue padded data
batched_data = padding_q.dequeue_many(5)

res = tf.contrib.learn.run_n({"z": batched_data}, n=1, feed_dict=None)

# Print the result
print("Batch shape: {}".format(res[0]["z"].shape))
print(res[0]["z"])