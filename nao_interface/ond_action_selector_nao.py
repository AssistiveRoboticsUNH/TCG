#!/usr/bin/env python

import os
import rospy
from nao_msgs.msg import WordRecognized
from std_msgs.msg import Int8
from model.TCG.TemporalContextGraph import TemporalContextGraph

word = ""


def word_callback(msg):
    global word
    word = msg.words[0]


def process_input(msg):
    global word
    ond_states = {'command': 0, 'prompt': 1, 'reward': 2, 'abort': 3}

    pub = rospy.Publisher("/action_msgs", Int8, queue_size=1)

    # Load Temporal Context Graph
    val_set_file = 'validation_set.txt'
    ond_path = os.path.join(os.getenv('HOME'), 'object_naming_dataset/')
    tcg = TemporalContextGraph(transition_events=['incorrect', 'correct', 'visual'])
    tcg.learn_model_from_files(os.path.join(ond_path, 'temp_info/'),
                               validation_file_path=os.path.join(ond_path, val_set_file))

    rospy.Subscriber('/word_recognized', WordRecognized, word_callback)
    print("CNN Packager ready")

    state = 'command'
    word = ''
    window_counter = 0
    tcg.initialize_policy_selector(state, 'abort')
    pub.publish(ond_states[state])
    start = rospy.get_rostime()
    while state not in ['abort', 'reward']:
        t = rospy.get_rostime() - start
        window_counter += 1
        if word == 'basketball':
            obs = 'correct'
            word = ''
        elif word == '':
            obs = 'none'
        else:
            obs = 'incorrect'
            word = ''
        tcg.process_observation(obs, t.to_sec())
        new_state = tcg.evaluate_timeout(t.to_sec())
        if new_state is not None:
            state = new_state
            pub.publish(ond_states[new_state])
            print('{}.{}   w:{} o:{}  s:{}'.format(t.secs, t.nsecs, window_counter, obs, state))
        else:
            print('{}.{}   w:{} o:{}'.format(t.secs, t.nsecs, window_counter, obs))
        rospy.sleep(1.0)
    print('Session completed.')


if __name__ == '__main__':
    sub = rospy.Subscriber("/start_request", Int8, process_input, )
    rospy.init_node("TCG_inference")
    print("TCG Inference ready")
    rospy.spin()
