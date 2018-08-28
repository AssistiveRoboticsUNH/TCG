#!/usr/bin/env python

import os
import rosbag

if __name__ == '__main__':
    zero_time = 0
    start_topic = '/sawyer_woz_action'
    end_topic = '/sawyer_msgs'
    human_topic = '/human_action'
    rosbag_dir = '/home/erp48/sawyer_bag/'
    for root, dirs, files in os.walk(rosbag_dir):
        for f in files:
            if '.bag' in f:
                out_file = f.replace('.bag', '.txt')
                print("Processing: {}".format(os.path.join(root, f)))
                with rosbag.Bag(os.path.join(root, f), 'r') as bag:
                    label_file = open(os.path.join(root, out_file), 'w+')
                    ps_counter = 0
                    pe_counter = 0
                    hs_counter = 0
                    he_counter = 0
                    for topic, msg, t in bag.read_messages():
                        if zero_time == 0:
                            zero_time = t
                        time = t - zero_time
                        str_time = "{}.{}".format(str(time.secs), str(time.nsecs).zfill(9))
                        if topic == start_topic:
                            if 1 <= msg.data <= 6:
                                label_file.write('pick_{}_s {}\n'.format(ps_counter, str_time))
                                ps_counter += 1
                            elif msg.data == 7:
                                label_file.write('place_dqa_s {}\n'.format(str_time))
                            elif msg.data == 8:
                                label_file.write('place_sqa_s {}\n'.format(str_time))
                            elif msg.data == 9:
                                label_file.write('place_box_s {}\n'.format(str_time))
                        elif topic == end_topic:
                            if 1 <= msg.data <= 6:
                                label_file.write('pick_{}_e {}\n'.format(pe_counter, str_time))
                                pe_counter += 1
                            elif msg.data == 7:
                                label_file.write('place_dqa_e {}\n'.format(str_time))
                            elif msg.data == 8:
                                label_file.write('place_sqa_e {}\n'.format(str_time))
                            elif msg.data == 9:
                                label_file.write('place_box_e {}\n'.format(str_time))
                        elif topic == human_topic:
                            if msg.data == 0:
                                label_file.write('ins_{}_s {}\n'.format(hs_counter, str_time))
                                hs_counter += 1
                            elif msg.data == 1:
                                label_file.write('ins_{}_e {}\n'.format(he_counter, str_time))
                                he_counter += 1
                    label_file.close()
                    zero_time = 0
