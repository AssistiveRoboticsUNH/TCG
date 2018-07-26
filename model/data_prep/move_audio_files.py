# Script used to move audio files to correct directories

from configuration import *


def move_audio_files():
    for dir, subdir, files in os.walk(ond_rosbags_path):
        subdir.sort()
        files.sort()
        for file in files:
            if file.endswith('.wav'):
                filepath = "{}/{}".format(dir, file)
                outdir = dir.replace('bags', 'audio_files')
                new_filepath = "{}/{}".format(outdir, file)

                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                print("Processing {}".format(filepath))
                os.rename(filepath, new_filepath)


if __name__ == "__main__":
    move_audio_files()
