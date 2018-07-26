# Script used to extract png frames from rosbags
# configured to extract frames where the subject may have left the intervention

import rosbag
import cv2
from cv_bridge import CvBridge
from configuration import *


def get_relevant_time_interval(filepath):
    start_time = 0
    end_time = 1000
    filepath = filepath.replace('bags', 'temp_info')
    filepath = filepath.replace('.bag', '.txt')
    f = open(filepath, 'r')
    for line in f:
        # if 'abort_e' in line:
        #     end_time = float(line.split(' ')[1])
        if 'prompt' in line and 's' in line:
            start_time = float(line.split(' ')[1]) - 2
    return start_time, end_time


def get_teleop_info_from_bags():
    bridge = CvBridge()
    for dir, subdir, files in os.walk(ond_rosbags_path):
        subdir.sort()
        files.sort()
        for file in files:
            if file.endswith('.bag') and 'failB' in dir:
                filepath = "{}/{}".format(dir, file)
                outdir = filepath.replace('bags', 'temp_pngs')
                outdir = outdir.replace('.bag', '')

                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                print("Processing {}".format(filepath))
                bag = rosbag.Bag(filepath)
                interval_start, interval_end = get_relevant_time_interval(filepath)
                start_time = None
                for topic, msg, t in bag.read_messages():
                    start_time = t
                    break
                for topic, msg, t in bag.read_messages(top_camera_topic):
                    if topic in top_camera_topic:
                        curr_time = t - start_time
                        if interval_start <= curr_time.secs <= interval_end:
                            curr_time = (curr_time.secs, curr_time.nsecs)
                            cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
                            image_name = "{}/{}_{}.png".format(outdir, curr_time[0], curr_time[1])
                            cv2.imwrite(image_name, cv_image)
                bag.close()


if __name__ == "__main__":
    # filepath = '/home/assistive-robotics/object_naming_dataset/bags/subject1/failB/fb_0.bag'
    # get_relevant_time_interval(filepath)
    get_teleop_info_from_bags()
