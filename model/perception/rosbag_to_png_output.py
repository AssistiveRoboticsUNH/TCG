#!/usr/bin/env python

# outputs the contents of a rosbag to a directory as png files

import heapq
import rospy
import rosbag

from model.perception.packager import *


topic_names = [
    '/action_finished',
    '/nao_robot/camera/top/camera/image_raw',
    '/audio/audio'
]


def read_timing_file(filename):
    # generate a heap of timing event tuples
    ifile = open(filename, 'r')
    timing_queue = []
    line = ifile.readline()
    while len(line) != 0:
        line = line.split()
        event_time = float(line[1])
        event_time = rospy.Duration(event_time)
        timing_queue.append((event_time, line[0]))
        line = ifile.readline()
    heapq.heapify(timing_queue)
    ifile.close()
    return timing_queue


def gen_TFRecord_from_file(bag_filename, timing_filename, flip=False):
    packager = CNNPackager(flip=flip)
    bag = rosbag.Bag(bag_filename)
    packager.p = False

    # parse timing file
    timing_queue = read_timing_file(timing_filename)
    # get first timing event
    current_time = heapq.heappop(timing_queue)
    timing_dict = {}

    all_timing_frames_found = False
    start_time = None

    for topic, msg, t in bag.read_messages(topics=topic_names):
        if start_time == None:
            start_time = t

        if not all_timing_frames_found and t > start_time + current_time[0]:
            # add the frame number anf timing label to frame dict
            timing_dict[current_time[1]] = packager.get_kinect_frame_count()
            if len(timing_queue) > 0:
                current_time = heapq.heappop(timing_queue)
            else:
                all_timing_frames_found = True
        if topic == topic_names[2]:
            packager.kinect_aud_callback(msg)

    # perform data pre-processing steps
    print('######## count: {}'.format(packager.get_kinect_frame_count()))
    packager.format_output()

    a = packager.get_kinect_aud_stack()
    a = np.reshape(a, [-1, aud_dtype["cmp_h"], aud_dtype["cmp_w"], aud_dtype["num_c"]])
    for i in range(a.shape[0]):
        a_img = show(a[i: i + 1], aud_dtype)
        a_img = cv2.resize(a_img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('/home/assistive-robotics/bag/audio/{}.jpg'.format(str(i + 1).zfill(3)), a_img)

    packager.reset()
    bag.close()


def show(data, d_type):
    tout = []
    out = []
    for i in range(data.shape[0]):
        imf = np.reshape(data[i], (d_type["cmp_h"], d_type["cmp_w"], d_type["num_c"]))

        limit_size = d_type["cmp_w"]
        frame_limit = 1

        if (d_type["cmp_w"] > limit_size):
            mod = limit_size / float(d_type["cmp_h"])
            imf = cv2.resize(imf, None, fx=mod, fy=mod, interpolation=cv2.INTER_CUBIC)

        if (imf.shape[2] == 2):
            imf = np.concatenate((imf, np.zeros((d_type["cmp_h"], d_type["cmp_w"], 1))),
                                 axis=2)
            imf[..., 0] = imf[..., 1]
            imf[..., 2] = imf[..., 1]
            imf = imf.astype(np.uint8)

        if (i % frame_limit == 0 and i != 0):
            if (len(tout) == 0):
                tout = out.copy()
            else:
                tout = np.concatenate((tout, out), axis=0)
            out = []
        if (len(out) == 0):
            out = imf
        else:
            out = np.concatenate((out, imf), axis=1)
    if (data.shape[0] % frame_limit != 0):
        fill = np.zeros((d_type["cmp_h"], d_type["cmp_w"] * (frame_limit -
                                                             (data.shape[0] % frame_limit)),
                         d_type["num_c"]))  # .fill(255)
        fill.fill(0)
        out = np.concatenate((out, fill), axis=1)
    if (len(out) != 0):
        if (len(tout) == 0):
            tout = out.copy()
        else:
            tout = np.concatenate((tout, out), axis=0)
        return tout


if __name__ == '__main__':
    rospy.init_node('gen_pngs_from_bag', anonymous=True)

    bag_file = os.environ["HOME"] + "/bag/6_emma+.bag"
    time_file = os.environ["HOME"] + "/bag/a0.txt"

    # generate a single file and store it as a scrap.tfrecord; Used for Debugging
    gen_TFRecord_from_file(bag_filename=bag_file, timing_filename=time_file, flip=False)
