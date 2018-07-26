# Script used to extract the temporal information of teleoperated actions from rosbags

import rosbag
from configuration import *


def format_action_name(action):
    if action == 'sd':
        return 'command'
    elif action == 'prompt':
        return 'prompt'
    elif action == 'reward':
        return 'reward'
    elif action == 'failure':
        return 'abort'
    elif action == 'stop':
        return 'stop'
    else:
        print("unhandled {}".format(action))
        return action


def get_teleop_info_from_bags():
    for dir, subdir, files in os.walk(ond_rosbags_path):
        subdir.sort()
        files.sort()
        for file in files:
            if file.endswith('.bag'):
                filepath = "{}/{}".format(dir, file)
                outdir = dir.replace('bags', 'temp_info')
                outfile = file.replace('.bag', '.txt')
                output_filepath = "{}/{}".format(outdir, outfile)

                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                outfile = open(output_filepath, 'w')

                print("Processing {}".format(filepath))
                bag = rosbag.Bag(filepath)
                start_time = None
                for topic, msg, t in bag.read_messages():
                    start_time = t
                    break
                for topic, msg, t in bag.read_messages(actions_topic):
                    if topic in actions_topic:
                        curr_time = t - start_time
                        curr_time = (curr_time.secs, curr_time.nsecs)
                        action = format_action_name(msg.action)
                        output = "{} {}.{}\n".format(action, curr_time[0], curr_time[1])
                        outfile.write(output)
                        print(output.replace('\n', ''))

                outfile.close()
                bag.close()


if __name__ == "__main__":
    get_teleop_info_from_bags()
