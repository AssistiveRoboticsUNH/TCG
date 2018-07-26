from datetime import datetime

from model.sg_perception.cnn_common_methods import *
from model.TCG.TemporalContextGraph import TemporalContextGraph

SEQUENCE_CHARS = ["^", "!", "_"]

WINDOW_SIZE = 20
WINDOW_STRIDE = 7

VALIDATION = True

OBS_TO_CLASS = {'abort': 'none',
                'command': 'none',
                'audio': 'response',
                'gesture': 'response',
                'prompt': 'none',
                'reward': 'none'}


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


def load_timing_dict(file):
    td = dict()
    with open(file, 'r') as infile:
        lines = infile.readlines()
        for line in lines:
            name, time = line.split(' ')
            td[name] = float(time)
    return td


if __name__ == '__main__':
    print("time start: {}".format(datetime.now()))

    sg_path = os.path.join(os.getenv('HOME'), 'social_greeting_dataset/')
    val_set_file = 'validation_set.txt'
    cnn_helper = CNNHelper(CLASSES_A, 'none', sg_path, val_set_file,
                           'ITBN_tfrecords/', WINDOW_SIZE, WINDOW_STRIDE)
    files_set = cnn_helper.get_files_set(validation=VALIDATION)
    print("Set size: {}".format(len(files_set)))

    # Load Temporal Context Graph
    tcg = TemporalContextGraph(transition_events=['response'])
    tcg.learn_model_from_files(os.path.join(sg_path, 'temp_info/'),
                               validation_file_path=os.path.join(sg_path, val_set_file))

    conf_matrix = np.zeros((len(CLASSES_A), len(CLASSES_A)), dtype=int)
    num_files = len(files_set)
    counter = 0
    fail_counter = 0
    sequences = dict()
    for f in files_set:
        if VALIDATION == ('validation' in f):
            counter += 1
            f = f.replace('ITBN_tfrecords', 'temp_info').replace(
                '.tfrecord', '.txt').replace('_validation', '')
            print("processing {}/{}: {}".format(counter, num_files, f))
            timing_dict = load_timing_dict(f)
            window_start, window_end, w_counter = cnn_helper.update_window_limits(-1)
            tcg.initialize_policy_selector('command', 'abort')
            real_sequence = ""
            pred_sequence = ""

            while tcg.sequence[-1].name != 'abort' and tcg.sequence[-1].name != 'reward':
                aud_label_data = cnn_helper.label_window(window_start / 10.0, window_end / 10.0, timing_dict)
                real_class = int(np.argmax(aud_label_data))
                selected_class = real_class
                tcg.process_observation(CLASSES_A[selected_class], window_end / 10.0)
                tcg.evaluate_timeout(window_end / 10.0)
                conf_matrix[real_class][selected_class] += 1
                real_sequence += SEQUENCE_CHARS[real_class]
                pred_sequence += SEQUENCE_CHARS[selected_class]
                window_start, window_end, w_counter = cnn_helper.update_window_limits(w_counter)
            if str([seq.name for seq in tcg.sequence]) != sequence_from_file(f):
                print([seq.name for seq in tcg.sequence])
                print(sequence_from_file(f))
                fail_counter += 1
            sequences[f] = real_sequence + "\n" + pred_sequence

    # for f in sorted(sequences.keys()):
    #     print("{}\n{}\n".format(f, sequences[f]))

    print("time end: {}\nfailures: {}\n".format(datetime.now(), fail_counter))
