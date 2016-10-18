import tensorflow as tf
import numpy as np
import tempfile

sequences = [[1, 2, 3], [4, 5, 1], [1, 2]]
label_sequences = [[0, 1, 0], [1, 0, 0], [1, 1]]

def make_example(sequence, labels):
    # The object we return
    ex = tf.train.SequenceExample()
    # A non-sequential feature of our example
    sequence_length = len(sequence)
    ex.context.feature["length"].int64_list.value.append(sequence_length)
    # Feature lists for the two sequential features of our example
    fl_tokens = ex.feature_lists.feature_list["tokens"]
    fl_labels = ex.feature_lists.feature_list["labels"]
    for token, label in zip(sequence, labels):
        fl_tokens.feature.add().int64_list.value.append(token)
        fl_labels.feature.add().int64_list.value.append(label)
    return ex

# Write all examples into a TFRecords file
# with tempfile.NamedTemporaryFile(delete=False) as fp:
    # writer = tf.python_io.TFRecordWriter(fp.name)
    # for sequence, label_sequence in zip(sequences, label_sequences):
    #     ex = make_example(sequence, label_sequence)
    #     writer.write(ex.SerializeToString())
    # writer.close()
    # print("Wrote to {}".format(fp.name))

tf.reset_default_graph()

# A single serialized example
# (You can read this from a file using TFRecordReader)
# ex = make_example([1, 2, 3], [0, 1, 0]).SerializeToString()

filename = "/tmp/tmpBiFt4P"
filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=None)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)


# Define how to parse the example
context_features = {
    "length": tf.FixedLenFeature([], dtype=tf.int64)
}
sequence_features = {
    "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
    "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
}

# Parse the example (returns a dictionary of tensors)
context_parsed, sequence_parsed = tf.parse_single_sequence_example(
    serialized=serialized_example,
    context_features=context_features,
    sequence_features=sequence_features
)

context = tf.contrib.learn.run_n(context_parsed, n=3, feed_dict=None)
print(context[1])
sequence = tf.contrib.learn.run_n(sequence_parsed, n=3, feed_dict=None)
print(sequence[1])

