import sys
from datetime import datetime

from aud_classifier import *
from model.perception.input_pipeline import *
from model.perception.cnn_common_methods import *

TAG = "aud_cnn"
ALPHA = 1e-5

WINDOW_SIZE = 15
WINDOW_STRIDE = 5

SEQUENCE_CHARS = ["^", "!", "_"]


if __name__ == '__main__':
    print("time start: {}".format(datetime.now()))

    ond_path = os.path.join(os.getenv('HOME'), 'object_naming_dataset/')
    val_set_file = 'validation_set.txt'
    cnn_helper = CNNHelper(CLASSES_A, 'none', ond_path, val_set_file,
                           'tfrecords/', WINDOW_SIZE, WINDOW_STRIDE)
    files_set = cnn_helper.get_files_set(validation=False)
    print("Training set size: {}".format(len(files_set)))

    # Load model
    cnn_chkpnt = "aud_cnn_final/model.ckpt"
    cnn = AudioCNNClassifier(learning_rate=ALPHA, filename=cnn_chkpnt)

    coord = tf.train.Coordinator()

    # read records from files into tensors
    seq_len_inp, opt_inp, aud_inp, timing_l_inp, timing_v_inp, name_inp = input_pipeline(files_set)

    # initialize all variables
    cnn.sess.run(tf.local_variables_initializer())
    cnn.sess.graph.finalize()
    threads = tf.train.start_queue_runners(coord=coord, sess=cnn.sess)

    print("Num Files: {}".format(len(files_set)))

    conf_matrix = np.zeros((len(CLASSES_A), len(CLASSES_A)), dtype=int)
    num_files = len(files_set)
    counter = 0
    sequences = dict()
    while len(files_set) > 0:
        # read a batch of tfrecords into np arrays
        dummy, aud_raw, timing_l, timing_v, name = cnn.sess.run([seq_len_inp, aud_inp, timing_l_inp,
                                                                 timing_v_inp, name_inp])
        tf_name = name[0].replace('.txt', '.tfrecord').replace('temp_info', 'tfrecords')
        if tf_name in files_set:
            counter += 1
            print("processing {}/{}: {}".format(counter, num_files, tf_name))
            files_set.remove(tf_name)
            timing_dict = parse_timing_dict(timing_l[0], timing_v[0])
            cnn_helper.adjust_timing_dict_for_audio(timing_dict, name[0])
            seq_len = len(aud_raw[0])
            window_start, window_end, w_counter = cnn_helper.update_window_limits(-1)
            real_sequence = ""
            pred_sequence = ""

            while window_end <= seq_len:
                # Label Data
                aud_label_data = cnn_helper.label_window(window_start, window_end, timing_dict)
                vals = {
                    cnn.seq_length_ph: dummy,
                    cnn.aud_ph: np.expand_dims(aud_raw[0][window_start:window_end], 0),
                    cnn.aud_y_ph: aud_label_data
                }
                aud_pred = cnn.sess.run([cnn.observe], feed_dict=vals)
                real_class = int(np.argmax(aud_label_data))
                selected_class = int(aud_pred[0][0])
                conf_matrix[real_class][selected_class] += 1
                real_sequence += SEQUENCE_CHARS[real_class]
                pred_sequence += SEQUENCE_CHARS[selected_class]
                window_start, window_end, w_counter = cnn_helper.update_window_limits(w_counter)
                if (window_end >= timing_dict.get('abort_e', sys.maxint) or
                   window_end >= timing_dict.get('reward_e', sys.maxint)):
                    break

            sequences[tf_name] = real_sequence + "\n" + pred_sequence

    for f in sorted(sequences.keys()):
        print("{}\n{}\n".format(f, sequences[f]))

    print("time end: {}\n{}\n".format(datetime.now(), conf_matrix))
