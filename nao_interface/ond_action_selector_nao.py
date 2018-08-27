#!/usr/bin/env python

import os
import rospy
from std_msgs.msg import Int8
from model.perception.constants import *
from model.TCG.TemporalContextGraph import TemporalContextGraph


def process_input(msg):
    ond_states = {'command': 0, 'prompt': 1, 'reward': 2, 'abort': 3}

    pub = rospy.Publisher("/action_msgs", Int8, queue_size=1)

    # Load Temporal Context Graph
    val_set_file = 'validation_set.txt'
    ond_path = os.path.join(os.getenv('HOME'), 'object_naming_dataset/')
    tcg = TemporalContextGraph(transition_events=['incorrect', 'correct', 'visual'])
    tcg.learn_model_from_files(os.path.join(ond_path, 'temp_info/'),
                               validation_file_path=os.path.join(ond_path, val_set_file))

    rospy.Subscriber('/audio/audio', AudioData, packager.kinect_aud_callback)
    print("CNN Packager ready")

    state = 'command'
    window_counter = 0
    tcg.initialize_policy_selector(state, 'abort')
    pub.publish(ond_states[state])
    start = rospy.get_rostime()
    while state not in ['abort', 'reward']:
        input_length = packager.get_kinect_frame_count()
        if input_length >= (WINDOW_STRIDE * window_counter + WINDOW_SIZE) * 2.8:
            start_frame = int(WINDOW_STRIDE * window_counter * 2.8)
            end_frame = int(start_frame + WINDOW_SIZE * 2.8)
            window_counter += 1
            # print('{}  {}  {}  {}'.format(window_counter, start_frame, end_frame, input_length))
            window_data = packager.format_kinect_range(start_frame, end_frame)
            obs = CLASSES_A[cnn.classify_input_ros(WINDOW_SIZE, window_data)]
            tcg.process_observation(obs, end_frame / 28.0)
            new_state = tcg.evaluate_timeout(end_frame / 28.0)
            # new_state = None

            t = rospy.get_rostime() - start
            if new_state is not None:
                state = new_state
                pub.publish(ond_states[new_state])
                print('{}.{}   w:{} o:{}  s:{}'.format(t.secs, t.nsecs, window_counter, obs, state))
            else:
                print('{}.{}   w:{} o:{}'.format(t.secs, t.nsecs, window_counter, obs))
    print('Session completed.')


if __name__ == '__main__':
    sub = rospy.Subscriber("/start_request", Int8, process_input, )
    rospy.init_node("TCG_inference")
    print("TCG Inference ready")
    rospy.spin()
