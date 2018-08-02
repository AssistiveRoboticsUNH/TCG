import sys
from datetime import datetime

from model.sg_perception.aud_classifier import aud_classifier
from model.sg_perception.opt_classifier import opt_classifier
from model.sg_perception.input_pipeline import *
from model.sg_perception.cnn_common_methods import *
from model.sg_perception.constants import *
from model.TCG.TemporalContextGraph import TemporalContextGraph

SEQUENCE_CHARS = ["^", "!", "_"]

ALPHA = 1e-5

WINDOW_SIZE = 20
WINDOW_STRIDE = 7

VALIDATION = True


def process_real_times(td):
    final_td = dict()
    delete_prompt = False
    mapping = {'noise_0_s': 'command_s', 'noise_0_e': 'command_e',
               'noise_1_s': 'prompt_s', 'noise_1_e': 'prompt_e',
               'audio_0_s': 'response_s', 'audio_0_e': 'response_e',
               'audio_1_s': 'response_s', 'audio_1_e': 'response_e',
               'gesture_0_s': 'response_s', 'gesture_0_e': 'response_e',
               'gesture_1_s': 'response_s', 'gesture_1_e': 'response_e'}
    if td.get('audio_0_s', None) is not None and td.get('audio_1_s', None) is not None:
        del td['audio_1_s']
        del td['audio_1_e']
        td['reward_s'] = td['noise_1_s']
        td['reward_e'] = td['noise_1_e']
        delete_prompt = True
    if td.get('gesture_0_s', None) is not None and td.get('gesture_1_s', None) is not None:
        del td['gesture_1_s']
        del td['gesture_1_e']
        td['reward_s'] = td['noise_1_s']
        td['reward_e'] = td['noise_1_e']
        delete_prompt = True
    if delete_prompt:
        del td['prompt_s']
        del td['prompt_e']
        del td['noise_1_s']
        del td['noise_1_e']
    if td.get('reward_s', None) is not None:
        mapping['abort_s'] = 'reward_s'
        mapping['abort_e'] = 'reward_e'
    for event, time in td.items():
        event = mapping.get(event, event)
        event_name = event.replace('_s', '').replace('_e', '')
        curr_time = final_td.get(event_name, (100000, -1))
        if '_s' in event:
            new_time = (min(time, curr_time[0]), curr_time[1])
        else:
            new_time = (curr_time[0], max(time, curr_time[1]))
        final_td[event_name] = new_time
    # for event in sorted(final_td):
    #     print('{}: {}'.format(event, final_td[event]))
    # print('DEBUG: {}')
    # for event in sorted(td):
    #     print('{}: {}'.format(event, td[event]))
    return final_td


def sequence_from_file(td):
    td = process_real_times(td)
    output = '['
    sequence = sorted(td.items(), key=lambda x: x[1])
    for item in sequence:
        item_str = "'{}', ".format(item[0].split('_')[0])
        output += item_str
    return output[:-2] + ']'


if __name__ == '__main__':
    print("time start: {}".format(datetime.now()))

    ond_path = os.path.join(os.getenv('HOME'), 'social_greeting_dataset/')
    val_set_file = 'validation_set.txt'
    cnn_helper = CNNHelper(CLASSES_A, 'none', ond_path, val_set_file,
                           'ITBN_tfrecords/', WINDOW_SIZE, WINDOW_STRIDE)
    files_set = cnn_helper.get_files_set(validation=VALIDATION)
    print("Set size: {}".format(len(files_set)))

    # Load Temporal Context Graph
    tcg = TemporalContextGraph(transition_events=['response'])
    tcg.learn_model_from_files(os.path.join(ond_path, 'temp_info/'),
                               validation_file_path=os.path.join(ond_path, val_set_file))

    # Load model
    a_ckpt = 'aud_classifier/aud_cnn_final/model.ckpt'
    o_ckpt = 'opt_classifier/opt_cnn_final/model.ckpt'
    aud_cnn = aud_classifier.ClassifierModel(learning_rate=ALPHA, filename=a_ckpt)
    opt_cnn = opt_classifier.ClassifierModel(learning_rate=ALPHA, filename=o_ckpt)

    # prepare tf objects
    aud_coord = tf.train.Coordinator()
    opt_coord = tf.train.Coordinator()

    # read records from files into tensors
    seq_len_inp, opt_raw_inp, aud_raw_inp, timing_labels_inp, timing_values_inp, file_name = \
        input_pipeline(files_set)

    # initialize variables
    with aud_cnn.sess.as_default():
        with aud_cnn.graph.as_default():
            aud_cnn.sess.run(tf.local_variables_initializer())
            aud_cnn.sess.graph.finalize()
            threads = tf.train.start_queue_runners(coord=aud_coord, sess=aud_cnn.sess)

    with opt_cnn.sess.as_default():
        with opt_cnn.graph.as_default():
            opt_cnn.sess.run(tf.local_variables_initializer())
            opt_cnn.sess.graph.finalize()
            threads = tf.train.start_queue_runners(coord=opt_coord, sess=opt_cnn.sess)

    conf_matrix = np.zeros((len(CLASSES_A), len(CLASSES_A)), dtype=int)
    num_files = len(files_set)
    counter = 0
    fail_counter = 0
    sequences = dict()
    while len(files_set) > 0:
        # read a batch of tfrecords into np arrays
        seq_len, opt_raw, aud_raw, timing_labels, timing_values, name = opt_cnn.sess.run(
            [seq_len_inp, opt_raw_inp, aud_raw_inp, timing_labels_inp, timing_values_inp,
             file_name])

        if VALIDATION:
            tf_name = name[0].replace('.txt', '_validation.tfrecord').replace(
                'PycharmProjects/dbn_arl/labels', 'social_greeting_dataset/ITBN_tfrecords')
        else:
            tf_name = name[0].replace('.txt', '.tfrecord').replace(
                'PycharmProjects/dbn_arl/labels', 'social_greeting_dataset/ITBN_tfrecords')

        if tf_name in files_set:
            counter += 1
            print("processing {}/{}: {}".format(counter, num_files, tf_name))
            files_set.remove(tf_name)
            if tf_name == '/home/assistive-robotics/social_greeting_dataset/ITBN_tfrecords/test_10/4_none1_validation.tfrecord':
                continue
            timing_dict = parse_timing_dict(timing_labels[0], timing_values[0])
            window_start, window_end, w_counter = cnn_helper.update_window_limits(-1)
            tcg.initialize_policy_selector('command', 'abort', delay=timing_dict['noise_0_s']/12.0)
            real_sequence = ""
            pred_sequence = ""

            while tcg.sequence[-1].name not in ['abort', 'reward']:
                # Label Data
                aud_label_data = cnn_helper.label_window(window_start, window_end, timing_dict)
                with aud_cnn.sess.as_default():
                    vals = {
                        aud_cnn.seq_length_ph: seq_len,
                        aud_cnn.aud_ph: np.expand_dims(aud_raw[0][window_start: window_end], 0),
                        aud_cnn.aud_y_ph: aud_label_data
                    }
                    aud_pred = aud_cnn.sess.run([aud_cnn.aud_observed], feed_dict=vals)
                    aud_selected_class = int(aud_pred[0][0])
                with opt_cnn.sess.as_default():
                    vals = {
                        opt_cnn.seq_length_ph: seq_len,
                        opt_cnn.pnt_ph: np.expand_dims(opt_raw[0][window_start: window_end], 0),
                        opt_cnn.pnt_y_ph: aud_label_data
                    }
                    opt_pred = opt_cnn.sess.run([opt_cnn.wave_observed], feed_dict=vals)
                    opt_selected_class = int(opt_pred[0][0])
                real_class = int(np.argmax(aud_label_data[0]))
                if max(opt_selected_class, aud_selected_class) == 2:
                    selected_class = 2
                else:
                    selected_class = 0
                tcg.process_observation(CLASSES_A[selected_class], window_end / 12.0)
                tcg.evaluate_timeout(window_end / 12.0)
                conf_matrix[real_class][selected_class] += 1
                real_sequence += SEQUENCE_CHARS[real_class]
                pred_sequence += SEQUENCE_CHARS[selected_class]
                window_start, window_end, w_counter = cnn_helper.update_window_limits(w_counter)
            timing_dict = process_real_times(timing_dict)
            if str([seq.name for seq in tcg.sequence]) != sequence_from_file(timing_dict):
                print([seq.name for seq in tcg.sequence])
                print(sequence_from_file(timing_dict))
                fail_counter += 1
            sequences[tf_name] = real_sequence + "\n" + pred_sequence

    for f in sorted(sequences.keys()):
        print("{}\n{}\n".format(f, sequences[f]))

    print("time end: {}\n{}\n".format(datetime.now(), conf_matrix))
    print("time end: {}\nfailures: {}\n".format(datetime.now(), fail_counter))
