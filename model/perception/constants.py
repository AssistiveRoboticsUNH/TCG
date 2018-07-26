INPUT_DIRECTORY = 'input/'

TOPIC_NAMES = ['/action_finished', '/nao_robot/camera/top/camera/image_raw',
               '/nao_robot/microphone/naoqi_microphone/audio_raw', '/audio/audio']

AUDIO_TOPICS = ['/nao_robot/microphone/naoqi_microphone/audio_raw', '/audio/audio']

VISUAL_SAMPLE_TOPICS = ['/nao_robot/camera/top/camera/image_raw',
                        '/nao_robot/microphone/naoqi_microphone/audio_raw', '/audio/audio']

AUD_GPU = '/gpu:0'
VID_GPU = '/gpu:0'

CLASSES_A = ["correct", "incorrect", "none"]
CLASSES_V = ["visual", "none"]

OBS_TO_CLASS = {'abort': 'none',
                'command': 'none',
                'correct': 'correct',
                'incorrect': 'incorrect',
                'prompt': 'none',
                'reward': 'none',
                'visual': 'visual'}

NUM_EPOCHS = 10000

img_h = 480
img_w = 640
img_size = 299
c_size = 64
aud_h = 128
aud_w = 8

img_dtype = {
    "name": "img",
    "img_h": img_h,
    "img_w": img_w,
    "num_c": 3,
    "cmp_h": img_size,
    "cmp_w": img_size
}

grs_dtype = {
    "name": "grs",
    "img_h": img_h,
    "img_w": img_w,
    "num_c": 1,
    "cmp_h": img_size,
    "cmp_w": img_size
}

opt_dtype = {
    "name": "pnt",
    "img_h": img_h,
    "img_w": img_w,
    "num_c": 1,
    "cmp_h": c_size,
    "cmp_w": c_size
}

aud_dtype = {
    "name": "aud",
    "img_h": img_h,
    "img_w": img_w,
    "num_c": 1,
    "cmp_h": aud_h,
    "cmp_w": aud_w
}
