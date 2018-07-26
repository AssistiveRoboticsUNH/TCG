from __future__ import print_function

import sys
from datetime import datetime

from vid_classifier import *
from model.perception.input_pipeline import *
from model.perception.cnn_common_methods import *

TAG = "opt_cnn"
ALPHA = 1e-5
NUM_ITER = 30000

WINDOW_SIZE = 20
WINDOW_STRIDE = 15

# Used to balance the number of training windows for each class. The order is the same as
# the CLASSES_V variable in constants.py
WINDOW_PROBABILITIES = [1.0, 0.0373]

if __name__ == '__main__':
    print("time start: {}".format(datetime.now()))

    ond_path = os.path.join(os.getenv('HOME'), 'object_naming_dataset/')
    val_set_file = 'validation_set.txt'
    cnn_helper = CNNHelper(CLASSES_V, 'none', ond_path, val_set_file,
                           'tfrecords/', WINDOW_SIZE, WINDOW_STRIDE)
    files_set = cnn_helper.get_files_set(validation=False)
    files_set.remove('/home/assistive-robotics/object_naming_dataset/tfrecords/subject2/failB/fb_3.tfrecord')

    # Generate Model
    cnn_chkpnt = ""
    cnn = VideoCNNClassifier(learning_rate=ALPHA, filename=cnn_chkpnt)

    # Train Model
    coord = tf.train.Coordinator()

    # read records from files into tensors
    seq_len_in, opt_in, aud_in, timing_l_in, timing_v_in, name_in = input_pipeline(files_set)

    # initialize all variables
    cnn.sess.run(tf.local_variables_initializer())
    cnn.sess.graph.finalize()
    threads = tf.train.start_queue_runners(coord=coord, sess=cnn.sess)

    print("Num Files: {},  Num iterations: {}".format(len(files_set), NUM_ITER))

    balance_check = [0, 0]
    last_dt = datetime.now()
    for iteration in range(NUM_ITER):
        # read a batch of tfrecords into np arrays
        dummy, opt_raw, timing_l, timing_v, name = cnn.sess.run([seq_len_in, opt_in, timing_l_in,
                                                                 timing_v_in, name_in])
        timing_dict = parse_timing_dict(timing_l[0], timing_v[0])
        seq_len = len(opt_raw[0])
        random_skew = np.random.randint(0, 5)
        window_start, window_end, w_counter = cnn_helper.update_window_limits(-1, random_skew)

        while window_end <= seq_len:
            window_rand_val = np.random.random_sample()
            opt_label_data = cnn_helper.label_window(window_start, window_end, timing_dict)
            window_class = int(np.argmax(opt_label_data))
            if window_rand_val < WINDOW_PROBABILITIES[window_class]:
                balance_check[window_class] += 1
                # Optimize Network
                vals = {
                    cnn.seq_length_ph: dummy,
                    cnn.opt_ph: np.expand_dims(opt_raw[0][window_start: window_end], 0),
                    cnn.opt_y_ph: opt_label_data
                }

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                _ = cnn.sess.run([cnn.optimizer_opt], feed_dict=vals,
                                 options=run_options, run_metadata=run_metadata)

            window_start, window_end, w_counter = cnn_helper.update_window_limits(w_counter,
                                                                                  random_skew)

            if (window_end >= timing_dict.get('abort_e', sys.maxint) or
                        window_end >= timing_dict.get('reward_e', sys.maxint)):
                break

        # Print Metrics
        if iteration % 100 == 0:
            past_dt = last_dt
            last_dt = datetime.now()
            print("iteration: {}\ttime: {}\tclass counts: {}".format(
                iteration, last_dt - past_dt, balance_check))

        # Delayed System Updates
        if iteration % 5000 == 0:
            # save the model to checkpoint file
            dir_name = TAG + "_" + str(iteration / 5000)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            cnn.save_model(save_dir=dir_name)

    # FINISH
    # save final model to checkpoint file
    dir_name = TAG + "_final"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    cnn.save_model(save_dir=dir_name)

    print("time end: {}".format(datetime.now()))
