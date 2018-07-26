#!/usr/bin/env python

import os
import rospy
import tensorflow as tf
from std_msgs.msg import Int8
from audio_common_msgs.msg import AudioData
from model.perception.constants import *
from model.perception.aud_classifier.aud_classifier import AudioCNNClassifier
from model.perception.packager import CNNPackager
from model.TCG.TemporalContextGraph import TemporalContextGraph

# cnn models paths
AUD_DQN_CHKPNT = "model/perception/aud_classifier/aud_cnn_final/model.ckpt"

# cnn parameters
ALPHA = 1e-5
WINDOW_SIZE = 15
WINDOW_STRIDE = 5


def process_input(msg):
    ond_states = {'command': 0, 'prompt': 1, 'reward': 2, 'abort': 3}

    pub = rospy.Publisher("/action_msgs", Int8, queue_size=1)

    # Load Temporal Context Graph
    val_set_file = 'validation_set.txt'
    ond_path = os.path.join(os.getenv('HOME'), 'object_naming_dataset/')
    tcg = TemporalContextGraph(transition_events=['incorrect', 'correct', 'visual'])
    tcg.learn_model_from_files(os.path.join(ond_path, 'temp_info/'),
                               validation_file_path=os.path.join(ond_path, val_set_file))

    # Load CNN model
    cnn_ckpt = "../model/perception/aud_classifier/aud_cnn_final/model.ckpt"
    cnn = AudioCNNClassifier(learning_rate=ALPHA, filename=cnn_ckpt)

    aud_coord = tf.train.Coordinator()

    # initialize variables
    cnn.sess.run(tf.local_variables_initializer())
    cnn.sess.graph.finalize()
    threads = tf.train.start_queue_runners(coord=aud_coord, sess=cnn.sess)

    packager = CNNPackager(input_dir='../model/perception/input/')
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
