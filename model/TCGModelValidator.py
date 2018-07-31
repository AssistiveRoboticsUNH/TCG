import sys
from datetime import datetime

from model.perception.aud_classifier.aud_classifier import AudioCNNClassifier
from model.perception.input_pipeline import *
from model.perception.cnn_common_methods import *
from model.perception.constants import *
from model.TCG.TemporalContextGraph import TemporalContextGraph

SEQUENCE_CHARS = ["^", "!", "_"]

ALPHA = 1e-5

WINDOW_SIZE = 15
WINDOW_STRIDE = 5

VALIDATION = True


def sequence_from_file(file):
    file_info = dict()
    temp_info = dict()
    cnn_helper.adjust_timing_dict_for_audio(file_info, file)
    for name, time in file_info.iteritems():
        start = '_s' in name
        name = name.replace('_s', '').replace('_e', '')
        saved_times = temp_info.get(name, (-1, -1))
        if start:
            temp_info[name] = (time, saved_times[1])
        else:
            temp_info[name] = (saved_times[0], time)
    sequence = sorted(temp_info.items(), key=lambda x: x[1])
    output = '['
    for item in sequence:
        item_str = "'{}', ".format(item[0].split('_')[0], item[1][0]/10.0, item[1][1]/10.0)
        output += item_str
    return output[:-2] + ']'


if __name__ == '__main__':
    print("time start: {}".format(datetime.now()))

    ond_path = os.path.join(os.getenv('HOME'), 'object_naming_dataset/')
    val_set_file = 'validation_set.txt'
    cnn_helper = CNNHelper(CLASSES_A, 'none', ond_path, val_set_file,
                           'tfrecords/', WINDOW_SIZE, WINDOW_STRIDE)
    files_set = cnn_helper.get_files_set(validation=VALIDATION)
    print("Set size: {}".format(len(files_set)))

    # Load Temporal Context Graph
    tcg = TemporalContextGraph(transition_events=['incorrect', 'correct', 'visual'])
    tcg.learn_model_from_files(os.path.join(ond_path, 'temp_info/'),
                               validation_file_path=os.path.join(ond_path, val_set_file))

    # Load model
    cnn_chkpnt = "perception/aud_classifier/aud_cnn_final/model.ckpt"
    cnn = AudioCNNClassifier(learning_rate=ALPHA, filename=cnn_chkpnt)

    coord = tf.train.Coordinator()

    # read records from files into tensors
    seq_len_inp, opt_inp, aud_inp, timing_l_inp, timing_v_inp, name_inp = input_pipeline(files_set)

    # initialize all variables
    cnn.sess.run(tf.local_variables_initializer())
    cnn.sess.graph.finalize()
    threads = tf.train.start_queue_runners(coord=coord, sess=cnn.sess)

    conf_matrix = np.zeros((len(CLASSES_A), len(CLASSES_A)), dtype=int)
    num_files = len(files_set)
    counter = 0
    fail_counter = 0
    sequences = dict()
    # files_set = ['/home/assistive-robotics/object_naming_dataset/tfrecords/subject5/failA/fa_4.tfrecord']
    while len(files_set) > 0:
        # read a batch of tfrecords into np arrays
        dummy, aud_raw, timing_l, timing_v, name = cnn.sess.run([seq_len_inp, aud_inp, timing_l_inp,
                                                                 timing_v_inp, name_inp])
        tf_name = name[0].replace('.txt', '.tfrecord').replace('temp_info', 'tfrecords')
        if 'failB' in tf_name and tf_name in files_set:
            files_set.remove(tf_name)
        elif tf_name in files_set:
            counter += 1
            print("processing {}/{}: {}".format(counter, num_files, tf_name))
            files_set.remove(tf_name)
            timing_dict = parse_timing_dict(timing_l[0], timing_v[0])
            cnn_helper.adjust_timing_dict_for_audio(timing_dict, name[0])
            seq_len = len(aud_raw[0])
            window_start, window_end, w_counter = cnn_helper.update_window_limits(-1)
            tcg.initialize_policy_selector('command', 'abort')
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
                tcg.process_observation(CLASSES_A[selected_class], window_end / 10.0)
                tcg.evaluate_timeout(window_end / 10.0)
                conf_matrix[real_class][selected_class] += 1
                real_sequence += SEQUENCE_CHARS[real_class]
                pred_sequence += SEQUENCE_CHARS[selected_class]
                window_start, window_end, w_counter = cnn_helper.update_window_limits(w_counter)
                if (window_end >= timing_dict.get('abort_e', sys.maxint) or
                   window_end >= timing_dict.get('reward_e', sys.maxint)):
                    break
            if str([seq.name for seq in tcg.sequence]) != sequence_from_file(name[0]):
                print([seq.name for seq in tcg.sequence])
                print(sequence_from_file(name[0]))
                fail_counter += 1
            sequences[tf_name] = real_sequence + "\n" + pred_sequence

    for f in sorted(sequences.keys()):
        print("{}\n{}\n".format(f, sequences[f]))

    print("time end: {}\n{}\n".format(datetime.now(), conf_matrix))
    print("time end: {}\nfailures: {}\n".format(datetime.now(), fail_counter))
