#!/usr/bin/env python

import os
import rospy
import tensorflow as tf
from std_msgs.msg import Int8
from model.sg_perception.constants import *
from model.sg_perception.aud_classifier import aud_classifier
from model.sg_perception.opt_classifier import opt_classifier
from model.sg_perception.packager import DQNPackager as CNNPackager
from model.TCG.TemporalContextGraph import TemporalContextGraph


# cnn parameters
ALPHA = 1e-5
WINDOW_SIZE = 20
WINDOW_STRIDE = 7

packager = None

def process_input(msg):
    sg_states = {'command': 0, 'prompt': 1, 'reward': 2, 'abort': 3}

    pub = rospy.Publisher("/asdpomdp/next_action", Int8, queue_size=1)

    # Load Temporal Context Graph
    val_set_file = 'validation_set.txt'
    sg_path = os.path.join(os.getenv('HOME'), 'social_greeting_dataset/')
    tcg = TemporalContextGraph(transition_events=['response'])
    tcg.learn_model_from_files(os.path.join(sg_path, 'temp_info/'),
                               validation_file_path=os.path.join(sg_path, val_set_file))
    tcg.nodes['reward'].duration = 1.2
    # Load CNN model
    a_cnn_ckpt = "../model/sg_perception/aud_classifier/aud_cnn_final/model.ckpt"
    v_cnn_ckpt = "../model/sg_perception/opt_classifier/opt_cnn_final/model.ckpt"
    aud_cnn = aud_classifier.ClassifierModel(learning_rate=ALPHA, filename=a_cnn_ckpt)
    opt_cnn = opt_classifier.ClassifierModel(learning_rate=ALPHA, filename=v_cnn_ckpt)

    # prepare tf objects
    aud_coord = tf.train.Coordinator()
    opt_coord = tf.train.Coordinator()

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

    print("Packager ready")

    state = 'command'
    window_counter = 0
    response_counter = 0
    packager.reset(hard_reset=True)
    tcg.initialize_policy_selector(state, 'abort')
    pub.publish(sg_states[state])
    start = rospy.get_rostime()
    while state not in ['abort']:
        if packager.get_frame_count() > (WINDOW_STRIDE * window_counter + WINDOW_SIZE):
            start_frame = WINDOW_STRIDE * window_counter
            end_frame = start_frame + WINDOW_SIZE
            window_counter += 1
            aud_window, opt_window = packager.formatOutput(start_frame=start_frame,
                                                           end_frame=end_frame)

            with aud_cnn.sess.as_default():
                obs_aud = aud_cnn.gen_prediction(WINDOW_SIZE, aud_window)
            with opt_cnn.sess.as_default():
                obs_opt = opt_cnn.gen_prediction(WINDOW_SIZE, opt_window)
            # obs_opt = obs_aud = 0

            print(packager.get_frame_count(), start_frame, end_frame, obs_aud, obs_opt)

            if obs_aud == 1:
                obs = 0
                response_counter = 0
            elif max(obs_aud, obs_opt) == 2:
                if response_counter == 1:
                    obs = 2
                    response_counter = 0
                else:
                    response_counter = 1
            else:
                obs = 0
                response_counter = 0
            tcg.process_observation(CLASSES_A[obs], end_frame / 12.0)
            new_state = tcg.evaluate_timeout(end_frame / 12.0)

            t = rospy.get_rostime() - start
            if new_state is not None:
                state = new_state
                pub.publish(sg_states[new_state])
                print('{}.{}   w:{} o:{}  s:{}'.format(t.secs, t.nsecs, window_counter, obs, state))
            else:
                print('{}.{}   w:{} o:{}'.format(t.secs, t.nsecs, window_counter, obs))
    print('Session completed.')


if __name__ == '__main__':
    sub = rospy.Subscriber("/start_request", Int8, process_input, )
    rospy.init_node("TCG_inference")
    packager = CNNPackager()
    print("TCG Inference ready")
    rospy.spin()
