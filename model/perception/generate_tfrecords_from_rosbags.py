# Madison Clark-Turner
# 12/2/2017
#
# Estuardo Carpio Mazariegos
# 5/30/2018

import heapq
import rospy
import rosbag

from tfrecord_rw import *
from packager import *


def read_temporal_info(filename):
    """
    Read the file containing the temporal information for a given rosbag
    :param filename: path to the file with the temporal information
    :return: a heap containing tuples with the temporal information of the events.
             The tuples are of the form (event_time, event_name)
    """
    file_obj = open(filename, 'r')
    timing_queue = []
    lines = file_obj.readlines()
    for line in lines:
        line = line.split()
        event_time = float(line[1])
        event_time = rospy.Duration.from_sec(event_time)
        timing_queue.append((event_time, line[0]))
    heapq.heapify(timing_queue)
    file_obj.close()
    return timing_queue


def gen_tfrecord_from_file(out_dir, out_filename, bag_filename, timing_filename,
                           flip=False, debug=False):
    """
    Generates the tfrecord file for a given rosbag
    :param out_dir: directory where the tfrecord file will be stored
    :param out_filename: name of the generated tfrecord file
    :param bag_filename: rosbag that will be used to generate the tfrecord file
    :param timing_filename: path to the file with the temporal information
    :param flip: boolean indicating if the image topics of the rosbag will be flipped horizontally
    :param debug: boolean indicating if debugging output will be generated by the packager
    :param generate_samples: boolean indicating if the visual samples for the training
                             set will be created
    """
    packager = CNNPackager(flip)
    bag = rosbag.Bag(bag_filename)
    events_temporal_info = read_temporal_info(timing_filename)
    current_time = heapq.heappop(events_temporal_info)
    temporal_info_dict = dict()

    temporal_info_complete = False
    start_time = rospy.Time(bag.get_start_time())
    for topic, msg, t in bag.read_messages(topics=TOPIC_NAMES):
        if not temporal_info_complete and t > start_time + current_time[0]:
            # add the frame number anf timing label to frame dict
            temporal_info_dict[current_time[1]] = packager.get_img_frame_count()
            if len(events_temporal_info) > 0:
                current_time = heapq.heappop(events_temporal_info)
            else:
                temporal_info_complete = True
        if topic == TOPIC_NAMES[1]:
            packager.img_callback(msg)
        elif topic == TOPIC_NAMES[2]:
            packager.nao_aud_callback(msg)
        elif topic == TOPIC_NAMES[3]:
            packager.kinect_aud_callback(msg)

    # perform data pre-processing steps
    packager.format_output(debug, output_dir + out_filename)

    # generate TFRecord data
    ex = make_sequence_example(packager.get_img_stack(), img_dtype,
                               packager.get_grs_stack(), grs_dtype,
                               packager.get_pnt_stack(), opt_dtype,
                               packager.get_nao_aud_stack(), aud_dtype,
                               packager.get_kinect_aud_stack(), aud_dtype,
                               temporal_info_dict, timing_filename)

    # write TFRecord data to file
    end_file = ".tfrecord"
    if flip:
        end_file = "_flip" + end_file

    writer = tf.python_io.TFRecordWriter(os.path.join(out_dir, out_filename + end_file))
    writer.write(ex.SerializeToString())
    writer.close()

    packager.reset()
    bag.close()


def generate_noise_sample_from_rosbags(bags_dir, temporal_info_dir, validation_set):
    """
    Creates a noise sample by appending together the first x seconds of each training sample. The
    value of x is set in the threshold_time variable and was determined by finding the largest value
    that would include only background noise for at least 95% of the training samples.
    :param bags_dir: directory containing the rosbags
    :param temporal_info_dir: directory containing the files with the temporal information
    :param validation_set: set of files that form the VALIDATION set
    """
    packager = CNNPackager()
    threshold_time = 1.6
    counter = 0
    for dir, subdir, files in os.walk(temporal_info_dir):
        subdir.sort()
        files.sort()
        for f in files:
            path = os.path.join(dir, f).replace('.txt', '.bag').replace(temporal_info_dir, bags_dir)
            if 'txt' in f and '~' not in f and path not in validation_set:
                duration = heapq.heappop(read_temporal_info(os.path.join(dir, f)))[0]
                time = float('{}.{}'.format(duration.secs, duration.nsecs))
                if time < threshold_time:
                    counter += 1
    threshold_time = rospy.Duration.from_sec(threshold_time)
    for dir, subdir, files in os.walk(bags_dir):
        subdir.sort()
        files.sort()
        for f in files:
            if '.bag' in f and f not in validation_set:
                bag = rosbag.Bag(os.path.join(dir, f))
                start_time = rospy.Time(bag.get_start_time())
                for topic, msg, t in bag.read_messages(AUDIO_TOPICS):
                    if t - start_time > threshold_time:
                        break
                    if topic == AUDIO_TOPICS[0]:
                        packager.nao_noise_callback(msg)
                    elif topic == AUDIO_TOPICS[1]:
                        packager.kinect_noise_callback(msg)
                bag.close()
    packager.save_noise_sample()
    packager.reset()


def generate_visual_sample_from_rosbags(bags_dir, validation_set):
    packager = CNNPackager(generate_samples=True)
    for dir, subdir, files in os.walk(bags_dir):
        subdir.sort()
        files.sort()
        for f in files:
            if '.bag' in f and f not in validation_set:
                print('Processing: {}'.format(os.path.join(dir, f)))
                bag = rosbag.Bag(os.path.join(dir, f))
                for topic, msg, t in bag.read_messages(VISUAL_SAMPLE_TOPICS):
                    if topic == VISUAL_SAMPLE_TOPICS[0]:
                        packager.img_callback(msg)
                    elif topic == VISUAL_SAMPLE_TOPICS[1]:
                        packager.nao_aud_callback(msg)
                    elif topic == VISUAL_SAMPLE_TOPICS[2]:
                        packager.kinect_aud_callback(msg)
                # perform data pre-processing steps
                packager.format_output(False, os.path.join(dir, f.replace('.bag', '')))
                bag.close()
                packager.reset()
    packager.save_visual_samples()
    packager.reset()


if __name__ == '__main__':
    rospy.init_node('gen_tfrecord')
    single_file_mode = True
    debug_output = True
    generate_noise_sample = False
    generate_visual_sample = False

    dataset_root_dir = "{}/object_naming_dataset/".format(os.environ["HOME"])
    bags_dir = dataset_root_dir + "bags/"
    temporal_info_dir = dataset_root_dir + "temp_info/"
    output_dir = dataset_root_dir + "tfrecords/"

    val_set = open(os.path.join(dataset_root_dir, 'validation_set.txt'))
    validation_set = list()
    for line in val_set.readlines():
        validation_set.append(line[:-1])
    val_set.close()

    if generate_noise_sample:
        generate_noise_sample_from_rosbags(bags_dir, temporal_info_dir, validation_set)
    if generate_visual_sample:
        generate_visual_sample_from_rosbags(bags_dir, validation_set)

    if single_file_mode:
        output_dir = output_dir.replace('tfrecords', 'tftest')
        subject = 'subject9'
        scenario = 'successC'
        bag = 'sc_1.bag'
        bag_path = '{}{}/{}/{}'.format(bags_dir, subject, scenario, bag)
        temporal_info = '{}{}/{}/{}'.format(temporal_info_dir, subject, scenario,
                                            bag.replace('.bag', '.txt'))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        gen_tfrecord_from_file(out_dir=output_dir, out_filename="test", bag_filename=bag_path,
                               timing_filename=temporal_info, flip=False, debug=debug_output)
    else:
        for root, subFolders, files in os.walk(bags_dir):
            subFolders.sort()
            files.sort()
            for f in files:
                if 'bag' in f:
                    bag_path = os.path.join(root, f)
                    tffiles_path = root.replace(bags_dir, output_dir)
                    tffile = f.replace('.bag', '')
                    temp_info = bag_path.replace(bags_dir, temporal_info_dir).replace(
                        '.bag', '.txt')
                    if not os.path.exists(tffiles_path):
                        os.makedirs(tffiles_path)
                    print('Generating: {}'.format(os.path.join(tffiles_path, tffile)))
                    gen_tfrecord_from_file(out_dir=tffiles_path, out_filename=tffile,
                                           bag_filename=bag_path, timing_filename=temp_info,
                                           flip=False, debug=debug_output)

    output_files = set()
    if debug_output:
        for root, subFolders, files in os.walk(output_dir):
            for f in files:
                if 'tfrecord' in f:
                    output_files.add(os.path.join(root, f))
    for f in output_files:
        coord = tf.train.Coordinator()
        filename_queue = tf.train.string_input_producer([f])
        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            threads = tf.train.start_queue_runners(coord=coord)

            def process_data(inp, data_type):
                data_s = tf.reshape(inp, [-1, data_type["cmp_h"],
                                          data_type["cmp_w"], data_type["num_c"]])
                return tf.cast(data_s, tf.uint8)

            context_parsed, sequence_parsed = parse_sequence_example(filename_queue)

            # information to extract from TFrecord
            seq_len = context_parsed["length"]
            top_img = process_data(sequence_parsed["top_img"], img_dtype)
            top_grs = process_data(sequence_parsed["top_grs"], grs_dtype)
            top_opt = process_data(sequence_parsed["top_opt"], opt_dtype)
            nao_aud = process_data(sequence_parsed["nao_aud"], aud_dtype)
            kinect_aud = process_data(sequence_parsed["kinect_aud"], aud_dtype)
            timing_labels = context_parsed["temporal_labels"]
            timing_values = sequence_parsed["temporal_values"]
            name = context_parsed["example_id"]

            l, i, g, o, na, ka, tl, tv, n = sess.run([seq_len, top_img, top_grs, top_opt,
                                                      nao_aud, kinect_aud, timing_labels,
                                                      timing_values, name])
            timing_dict = parse_timing_dict(tl, tv)

            coord.request_stop()
            coord.join(threads)
            sess.close()

            def show(data, d_type):
                tout = []
                out = []
                frame_limit = 12
                if d_type["name"] == "aud":
                    frame_limit = -1
                for i in range(data.shape[0]):
                    imf = np.reshape(data[i], (d_type["cmp_h"], d_type["cmp_w"], d_type["num_c"]))
                    limit_size = d_type["cmp_w"]
                    if d_type["cmp_w"] > limit_size:
                        mod = limit_size / float(d_type["cmp_h"])
                        imf = cv2.resize(imf, None, fx=mod, fy=mod, interpolation=cv2.INTER_CUBIC)

                    if imf.shape[2] == 2:
                        imf = np.concatenate((imf, np.zeros((d_type["cmp_h"], d_type["cmp_w"], 1))),
                                             axis=2)
                        imf[..., 0] = imf[..., 1]
                        imf[..., 2] = imf[..., 1]
                        imf = imf.astype(np.uint8)

                    if frame_limit > 0 and i % frame_limit == 0 and i != 0:
                        if len(tout) == 0:
                            tout = out.copy()
                        else:
                            tout = np.concatenate((tout, out), axis=0)
                        out = []
                    if len(out) == 0:
                        out = imf
                    else:
                        out = np.concatenate((out, imf), axis=1)
                if frame_limit > 0 and data.shape[0] % frame_limit != 0:
                    fill = np.zeros((d_type["cmp_h"], d_type["cmp_w"] *
                                     (frame_limit - (data.shape[0] % frame_limit)),
                                     d_type["num_c"]))
                    fill.fill(0)
                    out = np.concatenate((out, fill), axis=1)
                if len(out) != 0:
                    if len(tout) == 0:
                        tout = out.copy()
                    else:
                        tout = np.concatenate((tout, out), axis=0)
                    return tout

            # Use for visualizing Data Types
            show_from = 0
            # i_img = show(i[show_from:], img_dtype)
            # cv2.imwrite(f.replace('.tfrecord', '_i.jpg'), i_img)
            # os.system('gnome-open {}'.format(f.replace('.tfrecord', '_o.jpg'))

            # g_img = show(g[show_from:], grs_dtype)
            # cv2.imwrite(f.replace('.tfrecord', '_g.jpg'), g_img)

            o_img = show(o[show_from:], opt_dtype)
            cv2.imwrite(f.replace('.tfrecord', '_o.jpg'), o_img)

            na_img = show(na[show_from:], aud_dtype)
            cv2.imwrite(f.replace('.tfrecord', '_sna.jpg'), na_img)

            ka_img = show(ka[show_from:], aud_dtype)
            cv2.imwrite(f.replace('.tfrecord', '_ska.jpg'), ka_img)
