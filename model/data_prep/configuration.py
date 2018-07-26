# File containing configurations used in the scripts and programs contained in this package

import os

# Path that contains the bag files for the object naming dataset
ond_rosbags_path = '{}/{}'.format(os.getenv("HOME"), 'object_naming_dataset')

# Name of the topic containing the teleoperated actions performed during the interventions
actions_topic = '/action_started'

# Top camera topic
top_camera_topic = '/nao_robot/camera/top/camera/image_raw'
