from model.perception.constants import *
from tfrecord_rw import *


def input_pipeline(input_files):
    files_queue = tf.train.string_input_producer(input_files, num_epochs=NUM_EPOCHS, shuffle=True)

    min_after_dequeue = 7  # buffer to shuffle with (bigger=better shuffling)
    capacity = min_after_dequeue + 3

    context_parsed, sequence_parsed = parse_sequence_example(files_queue)

    seq_len = context_parsed["length"]
    temporal_labels = context_parsed["temporal_labels"]
    temporal_values = sequence_parsed["temporal_values"]
    name = context_parsed["example_id"]

    def process_data(inp, data_type):
        data_s = tf.reshape(inp, [-1, data_type["cmp_h"] * data_type["cmp_w"] * data_type["num_c"]])
        return tf.cast(data_s, tf.uint8)

    opt_raw = process_data(sequence_parsed["top_grs"], grs_dtype)
    aud_raw = process_data(sequence_parsed["kinect_aud"], aud_dtype)

    NUM_THREADS = 1

    inputs = [seq_len, opt_raw, aud_raw, temporal_labels, temporal_values, name]

    dtypes = list(map(lambda x: x.dtype, inputs))
    shapes = list(map(lambda x: x.get_shape(), inputs))

    queue = tf.RandomShuffleQueue(capacity, min_after_dequeue, dtypes)

    enqueue_op = queue.enqueue(inputs)
    qr = tf.train.QueueRunner(queue, [enqueue_op] * NUM_THREADS)

    tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, qr)
    inputs = queue.dequeue()

    for tensor, shape in zip(inputs, shapes):
        tensor.set_shape(shape)

    in_fields = tf.train.batch(inputs, 1, capacity=capacity, dynamic_pad=True)

    return in_fields[0], in_fields[1], in_fields[2], in_fields[3], in_fields[4], in_fields[5]
