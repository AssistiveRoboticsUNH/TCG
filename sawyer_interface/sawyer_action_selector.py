#!/usr/bin/env python

import os
import rospy
from std_msgs.msg import Int8, Bool
from TCG.TemporalContextGraph import TemporalContextGraph, TCGNode

obs = 0


def process_input(msg):
    global obs
    tcg_states = {'abort': 0, 'pick': 6, 'dqa': 7, 'sqa': 8, 'box': 9}

    pub = rospy.Publisher("/tcg_msgs", Int8, queue_size=1)

    # Load Temporal Context Graph
    val_set_file = 'validation.txt'
    dataset_path = os.path.join(os.getenv('HOME'), 'sawyer_bag/')
    tcg = TemporalContextGraph(transition_events=['ins'])
    tcg.learn_model_from_files(os.path.join(dataset_path, 'temp_info/'),
                               validation_file_path=os.path.join(dataset_path, val_set_file))
    tcg.output_graph('output/sawyer')

    state = 'pick'
    tcg.nodes['abort'] = TCGNode('abort', False, True, False, 1, 1, None)
    tcg.initialize_policy_selector(state, 'abort')
    pub.publish(tcg_states[state])
    tcg_states[state] = tcg_states[state] - 1
    start = rospy.get_rostime()
    while state not in ['abort']:
        time = rospy.get_rostime() - start
        tcg.process_observation(obs, time.to_sec())
        new_state = tcg.evaluate_timeout(time.to_sec())
        if new_state is not None:
            state = new_state
            pub.publish(tcg_states[new_state])
            print('{}.{}   o:{}  s:{}'.format(time.secs, time.nsecs, obs, state))
        else:
            print('{}.{}   o:{}'.format(time.secs, time.nsecs, obs))
        rospy.sleep(0.5)
    print('Session completed.')


def process_obs(msg):
    global obs
    if msg.data == 0:
        obs = 'ins'
    elif msg.data == 1:
        obs = 'idle'


if __name__ == '__main__':
    sub = rospy.Subscriber("/run_sawyer_auto", Bool, process_input, )
    sub_obs = rospy.Subscriber("/human_action", Int8, process_obs, )
    rospy.init_node("TCG_inference")
    print("TCG Inference ready")
    rospy.spin()
