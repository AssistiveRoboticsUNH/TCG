# Script used to find the optimal values for the CNN parameters FRAME_SIZE and STRIDE
# The output also includes a total count of windows for each class. This values are used to
# skip some training windows, thus balancing the number of windows used for training for each class

import sys
from datetime import datetime

from model.perception.tfrecord_rw import *
from model.perception.cnn_common_methods import *

FRAME_SIZE = 20
STRIDE = 5
CLASSES = CLASSES_A
DEFAULT_CLASS = 'none'
GPU = '/gpu:0'


def process_data(inp, data_type):
    """
    Reshapes data retrieved from a tfrecord file
    :param inp: vector containing the input data
    :param data_type: dict containing the data type information
    :return: reshaped vector
    """
    data_s = tf.reshape(inp, [-1, data_type["cmp_h"], data_type["cmp_w"], data_type["num_c"]])
    return tf.cast(data_s, tf.uint8)


if __name__ == '__main__':
    print("time start: {}".format(datetime.now()))

    # generate list of input files
    dataset_base_dir = os.path.join(os.getenv('HOME'), 'object_naming_dataset')
    validation_file = 'validation_set.txt'
    tf_records_dir = 'tfrecords/'
    cnn_helper = CNNHelper(CLASSES, DEFAULT_CLASS, dataset_base_dir, validation_file,
                           tf_records_dir, FRAME_SIZE, STRIDE)
    files_set = cnn_helper.get_files_set(validation=False)
    print("Training set size: {}".format(len(files_set)))

    balance_check = np.zeros((len(CLASSES)), dtype=int)
    last_dt = datetime.now()
    coord = None
    threads = None
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        coord = tf.train.Coordinator()
        filename_queue = tf.train.string_input_producer(files_set, num_epochs=1, shuffle=False)
        for count, f in enumerate(files_set):
            sess.run(tf.local_variables_initializer())
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            context_parsed, sequence_parsed = parse_sequence_example(filename_queue)
            kinect_aud = process_data(sequence_parsed["kinect_aud"], aud_dtype)
            timing_labels = context_parsed["temporal_labels"]
            timing_values = sequence_parsed["temporal_values"]
            name = context_parsed["example_id"]

            with tf.device(GPU):
                ka, tl, tv, n = sess.run([kinect_aud, timing_labels, timing_values, name])
            seq_len = len(ka)
            timing_dict = parse_timing_dict(tl, tv)
            cnn_helper.adjust_timing_dict_for_audio(timing_dict, n)
            start_frame, end_frame, w_counter = cnn_helper.update_window_limits(-1)

            while end_frame <= seq_len:
                aud_label_data = cnn_helper.label_window(start_frame, end_frame, timing_dict)
                window_class = int(np.argmax(aud_label_data))
                balance_check[window_class] += 1
                start_frame, end_frame, w_counter = cnn_helper.update_window_limits(w_counter)
                if (end_frame >= timing_dict.get('abort_e', sys.maxint) or
                   end_frame >= timing_dict.get('reward_e', sys.maxint)):
                    break

            print("{} - Class count: [c, i, n] {} {}".format(count, balance_check, n))
        coord.request_stop()
        coord.join(threads)

    total_windows = float(min(balance_check))
    print("balance coef: {}".format([total_windows / val for val in balance_check]))
    print("time end: {}".format(datetime.now()))
