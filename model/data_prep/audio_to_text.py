# file created to get the audio of the sessions using google cloud services.
# the bucket name needs to match the bucket created in google cloud.
# Files with no recognized audio will not generate an output file

# this file needs the env. var GOOGLE_APPLICATION_CREDENTIALS to point to a valid gc key

from configuration import *

bucket = 'gs://temp_ond/'
local_bucket = '/home/assistive-robotics/object_naming_dataset/'


def get_transcripts():
    # Imports the Google Cloud client library
    from google.cloud import speech
    from google.cloud.speech import enums
    from google.cloud.speech import types

    # Instantiates a client
    client = speech.SpeechClient()

    # Transcribe audio files
    responses = dict()
    for root, subFolders, files in os.walk(ond_rosbags_path):
        for file_name in files:
            if '.wav' in file_name:
                file_path = root + "/" + file_name
                file_uri = file_path.replace(local_bucket, bucket)
                print('Processing: {}'.format(file_uri))

                audio = types.RecognitionAudio(uri=file_uri)
                config = types.RecognitionConfig(
                    encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code='en-US',
                    enable_word_time_offsets=True)

                # Detects speech in the audio file
                responses[file_path] = client.long_running_recognize(config, audio)

    for file_name, future in responses.items():
        out_name = file_name.replace(".wav", ".txt")
        print('Waiting for: {}'.format(out_name))
        try:
            response = future.result(timeout=10000)
            with open(out_name, "w") as out_file:
                out_file.write(file_name + "\n" + str(response))
                print('Completed: {}'.format(out_name))
        except:
            print('Failed: {}'.format(out_name))


if __name__ == '__main__':
    get_transcripts()


# did not generate output file:
# /home/assistive-robotics/object_naming_dataset/audio_files/subject4/failB/fb_2.txt
# /home/assistive-robotics/object_naming_dataset/audio_files/subject8/failB/fb_2.txt
# /home/assistive-robotics/object_naming_dataset/audio_files/subject6/failB/fb_3.txt
