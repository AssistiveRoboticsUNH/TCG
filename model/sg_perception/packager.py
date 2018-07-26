# dqn_packager_itbn.py
# Madison Clark-Turner
# 12/2/2017

import math
import threading
import matplotlib.pyplot as plt

# image pre-processing and optical flow generation
import cv2
import librosa
import librosa.display
import numpy as np

# ROS
import rospy
from cv_bridge import CvBridge

# audio pre-processing
from nao_msgs.msg import AudioBuffer
from scipy import signal
from sensor_msgs.msg import Image
from std_msgs.msg import Int8

from model.perception.noise_subtraction import reduce_noise
from model.sg_perception.constants import *

topic_names = [
    '/action_finished',
    '/nao_robot/camera/top/camera/image_raw',
    '/nao_robot/microphone/naoqi_microphone/audio_raw'
]
NOISE_SAMPLE_PATH = '../model/sg_perception/input/noise#.npy'
WAVE_SAMPLE_PATH = '../model/sg_perception/input/wave_reference_frame.npy'
HAAR_CASCADE = '../model/sg_perception/input/haarcascade_frontalface_default.xml'


'''
DQN Packager listens to the topics for images and audio.
Processes those inputs into sequences and passes the result to 
the DQN model.
'''


class DQNPackager:
    def __init__(self, aud_classifier=None, opt_classifier=None, aud_frame_size=20, aud_stride=7,
                 opt_frame_size=20, opt_stride=7, flip=False):
        # dqn model
        self.__flip = flip
        self.__noise_sample_1 = np.load(NOISE_SAMPLE_PATH.replace('#', '1'))
        self.__noise_sample_2 = np.load(NOISE_SAMPLE_PATH.replace('#', '2'))
        self.__noise_dummy = np.load(NOISE_SAMPLE_PATH.replace('#', '3'))
        self.__wave_sample = np.load(WAVE_SAMPLE_PATH)
        self.p = 0
        self.frame_counter = 0

        self.debug_times = dict()
        self.debug_times['var'] = 0
        self.debug_times['format'] = 0
        self.debug_times['acnn'] = 0
        self.debug_times['ocnn'] = 0
        self.debug_times['itbn'] = 0
        self.debug_times['total'] = 0

        self.__aud_complete_stack = 0
        self.__img_complete_stack = 0
        self.__audStack = 0
        self.__imgStack = 0

        # variables for tracking images received
        self.__most_recent_act = -1
        self.__lock = threading.Lock()
        self.reset()

        # variables for optical flow
        self.frame1 = []
        self.prvs, self.hsv = None, None

        # variables for audio
        self.counter = 0
        self.__face_cascade = cv2.CascadeClassifier(HAAR_CASCADE)
        self.rate = 16000  # Sampling rate

        # subscribers
        QUEUE_SIZE = 100
        self.sub_act = rospy.Subscriber(topic_names[0],
                                        Int8, self.actCallback, queue_size=QUEUE_SIZE)
        self.sub_img = rospy.Subscriber(topic_names[1],
                                        Image, self.imgCallback, queue_size=QUEUE_SIZE)
        self.sub_aud = rospy.Subscriber(topic_names[2],
                                        AudioBuffer, self.audCallback, queue_size=QUEUE_SIZE)

        self.init_time = None

    def getRecentAct(self):
        return self.__most_recent_act

    def getImgStack(self):
        return self.__imgStack

    def getPntStack(self):
        return self.__pntStack

    def getAudStack(self):
        return self.__specStack

    ############################
    # Collect Data into Frames #
    ############################

    def setPrint(self, p):
        self.p = p

    def clearMsgs(self):
        self.__recent_msgs = [False] * 2

    def reset(self, reset_time=-1, already_locked=False, hard_reset=False):
        if (not already_locked):
            self.__lock.acquire()

        self.clearMsgs()
        self.save_partial_input()
        self.__imgStack = 0
        self.__pntStack = 0
        self.__audStack = 0
        self.count = {"img": 0, "aud": 0}

        self.frame1 = []
        self.prvs, self.hsv = None, None

        if hard_reset:
            self.__aud_complete_stack = 0
            self.__img_complete_stack = 0
            self.init_time = rospy.get_rostime()
            self.frame_counter = 0

        if (not already_locked):
            self.__lock.release()

    def actCallback(self, msg):
        self.__most_recent_act = msg.data
        return

    def imgCallback(self, msg):
        self.__recent_msgs[0] = msg
        self.checkMsgs("img")
        return

    def audCallback(self, msg):
        self.__recent_msgs[1] = msg
        self.checkMsgs("aud")
        return

    def formatAudMsg(self, aud_msg):
        # shapes the audio file for use later
        data = aud_msg.data
        data = np.reshape(data, (-1, 4))
        data = data.transpose([1, 0])
        return data[0]

    def checkMsgs(self, src):
        # may need to use mutexes on self.__recent_msgs
        self.__lock.acquire()
        if False in self.__recent_msgs:
            self.count[src] += 1
            self.__lock.release()
            return

        if (self.p):
            print("FRAME ADDED!")
        # organize and send data
        img = self.__recent_msgs[0]
        aud = self.__recent_msgs[1]

        if type(self.__imgStack) == int:
            self.__imgStack = [img]
            self.__audStack = [aud]
        else:
            self.__imgStack.append(img)
            self.__audStack.append(aud)
        self.frame_counter += 1
        self.clearMsgs()
        self.__lock.release()

    def getFrameCount(self):
        if (type(self.__imgStack) == int):
            return 0
        return len(self.__imgStack)

    ###############
    # Format Data #
    ###############

    def formatImgBatch(self, img_stack, name=""):
        # pre-process the RGB input and generate the optical flow
        img_out, pnt_out = [], []
        # max_mean = -1
        # max_index = -1
        # index = 0
        for x in img_stack:
            img = self.formatImg(x)
            img_out.append(np.asarray(img).flatten())

            opt_flow = self.formatOpt(img)
            pnt_out.append(opt_flow)
            # if max_mean < np.ndarray.mean(opt_flow):
            #     max_mean = np.ndarray.mean(opt_flow)
            #     max_index = index
            # index += 1
        # np.save(WAVE_SAMPLE_PATH, pnt_out[max_index])
        pnt_out.append(self.__wave_sample)
        pnt_out = cv2.normalize(np.asarray(pnt_out), None, 0, 255, cv2.NORM_MINMAX)
        pnt_out = pnt_out[:-1]

        for opt in pnt_out:
            opt = np.asarray(opt).flatten()

        return img_out, pnt_out

    def formatImg(self, img_msg):
        # pre-process the image data to crop it to an appropriate size

        # convert image to cv2 image
        img = CvBridge().imgmsg_to_cv2(img_msg, "bgr8")

        # identify location of face if possible using Haarcascade and
        # crop to center on it.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.__face_cascade.detectMultiScale(gray, 1.3, 5)
        x, y, w, h = -1, -1, -1, -1

        # if face is locataed to the edge set then crop from the opposite side
        buff = img.shape[1] - img.shape[0]

        if (len(faces) > 0):
            for (xf, yf, wf, hf) in faces:
                if wf * hf > w * h:
                    x, y, w, h = xf, yf, wf, hf

            if (x >= 0 and y >= 0 and x + w < 320 and y + h < 240):
                y, h = 0, img.shape[0]
                mid = x + (w / 2)
                x, w = mid - (img.shape[0] / 2), img.shape[0]
                if (x < 0):
                    x = 0
                elif (x > buff):
                    x = buff / 2
                img = img[y:y + h, x:x + w]
        else:
            # if no face visible set crop image to center of the video
            diff = img.shape[1] - img.shape[0]
            img = img[0:img.shape[0], (diff / 2):(img.shape[1] - diff / 2)]

        # resize image to 299 x 299
        y_mod = 1 / (img.shape[0] / float(img_dtype["cmp_h"]))
        x_mod = 1 / (img.shape[1] / float(img_dtype["cmp_w"]))
        img = cv2.resize(img, None, fx=x_mod, fy=y_mod, interpolation=cv2.INTER_CUBIC)

        if (self.__flip):
            # if flip set to true then mirror the image horizontally
            img = np.flip(img, 1)

        return img

    '''
    def formatOpt(self, img_src):
        # generate optical flow
        img = img_src.copy()    
        t = rospy.get_rostime()
        
        
        opt_img = np.zeros(img.shape)[..., 0]
        
        if(len(self.frame1)==0):
            # if img is the first frame then set optical flow to be black screen
            self.frame1 = img
            self.prvs = cv2.cvtColor(self.frame1,cv2.COLOR_BGR2GRAY)    
        else:
            frame2 = img    
            # generate optical flow
            #print(self.prvs [0][0])
            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(self.prvs,next, 0.5, 1, 2, 5, 7, 1.5, 1)    
            t = rospy.get_rostime()
            opt_img, ang = cv2.cartToPolar(flow[...,0], flow[...,1])    
            # normalize the magnitude to between 0 and 255 (replace with other normalize to prevent precission issues)
            opt_img = cv2.normalize(opt_img,None,0,255,cv2.NORM_MINMAX) #<-- if there are issues see if using this normalize fixes them
            self.prvs = next    
        mod = pnt_dtype["cmp_h"]/float(img_dtype["cmp_h"])
        
        opt_img = cv2.resize(opt_img,None,fx=mod, fy=mod, interpolation=cv2.INTER_CUBIC)    
        
        #opt_img = cv2.normalize(opt_img,None,0,255,cv2.NORM_MINMAX)
        
        return np.asarray(opt_img).flatten()
    '''

    def formatOpt(self, img_src):
        # generate optical flow
        img = img_src.copy()
        mod = pnt_dtype["cmp_h"] / float(img_dtype["cmp_h"])

        img = cv2.resize(img, None, fx=mod, fy=mod, interpolation=cv2.INTER_CUBIC)
        opt_img = np.zeros(img.shape)[..., 0]

        if (len(self.frame1) == 0):
            # if img is the first frame then set optical flow to be black screen
            self.frame1 = img
            self.prvs = cv2.cvtColor(self.frame1, cv2.COLOR_BGR2GRAY)
        else:
            frame2 = img

            # generate optical flow
            # print(self.prvs [0][0])
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            # flow = cv2.calcOpticalFlowFarneback(self.prvs,next, 0.5, 1, 2, 5, 7, 1.5, 1)
            flow = cv2.calcOpticalFlowFarneback(self.prvs, next, None, 0.5, 1, 8, 10, 7, 1.5, 1)

            opt_img, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # normalize the magnitude to between 0 and 255 (replace with other normalize to prevent precission issues)
            # opt_img = cv2.normalize(opt_img,None,0,255,cv2.NORM_MINMAX) #<-- if there are issues see if using this normalize fixes them
            self.prvs = next

        return opt_img

    def formatAudBatch(self, aud_msg_array, name=""):
        # perform pre-processing on the audio input

        for x in range(len(aud_msg_array)):
            aud_msg_array[x] = self.formatAudMsg(aud_msg_array[x])

        num_frames = len(aud_msg_array)
        core_data = np.reshape(aud_msg_array, (num_frames * len(aud_msg_array[0])))
        # modify data
        # core_data = input_data
        # core_data = input_data[:int(16000*1.2)]

        # np.save(NOISE_SAMPLE_PATH.replace('#', '3'), core_data)
        # dummy = np.load(NOISE_SAMPLE_PATH.replace('#', '3'))
        # core_data = np.append(core_data, dummy)
        # num_frames = num_frames + int(len(dummy)/len(aud_msg_array[0]))

        # get the indicies for the noise sample
        # noise_sample_s, noise_sample_e = 1, -1 #16000 * (-1.5), -1

        # perform spectral subtraction to reduce noise
        noise = self.__noise_sample_1
        # noise = core_data[int(noise_sample_s): int(noise_sample_e)]
        # np.save(NOISE_SAMPLE_PATH.replace('#', '1'), noise)
        filtered_input = reduce_noise(np.array(core_data), noise)

        # smooth signal
        b, a = signal.butter(3, 0.05)
        filtered_input = signal.lfilter(b, a, filtered_input)

        # additional spectral subtraction to remove remaining noise
        noise = self.__noise_sample_2
        # noise = filtered_input[int(noise_sample_s): int(noise_sample_e)]
        # np.save(NOISE_SAMPLE_PATH.replace('#', '2'), noise)
        filtered_input = reduce_noise(filtered_input, noise)

        filtered_input = np.append(filtered_input, self.__noise_dummy)
        num_frames = num_frames + int(len(self.__noise_dummy)/len(aud_msg_array[0]))

        # generate spectrogram
        S = librosa.feature.melspectrogram(y=filtered_input, sr=self.rate, n_mels=128, fmax=8000)
        S = librosa.power_to_db(S, ref=np.max)

        # if(True):
        #     # if True then output spectrogram to png file (requires matplot.pyplot lib to be imported)
        #     plt.figure(figsize=(10, 4))
        #
        #     librosa.display.specshow(S,y_axis='mel', fmax=8000,x_axis='time')
        #     plt.colorbar(format='%+2.0f dB')
        #     plt.title('Mel-Spectrogram')
        #     plt.tight_layout()
        #     print("spectrogram ouput to file.")
        #
        #     out_file = "debug/audio_{}.png".format(self.__chunk_counter)
        #     plt.savefig(out_file)
        #     self.counter += 1
        #     plt.clf()

        # split the spectrogram into A_i. This generates an overlap between
        # frames with as set stride
        stride = S.shape[1] / float(num_frames)
        frame_len = aud_dtype["cmp_w"]

        # pad the entire spectrogram so that overlaps at either end do not fall out of bounds
        min_val = np.nanmin(S)

        empty = np.zeros((S.shape[0], 3))
        empty.fill(min_val)
        empty_end = np.zeros((S.shape[0], 8))
        empty_end.fill(min_val)
        S = np.concatenate((empty, S, empty_end), axis=1)

        split_data = np.zeros(shape=(num_frames, S.shape[0], frame_len), dtype=S.dtype)
        for i in range(0, num_frames):
            split_data[i] = S[:,
                            int(math.floor(i * stride)):int(math.floor(i * stride)) + frame_len]

        # normalize the output to be between 0 and 255
        split_data -= split_data.min()
        split_data                 /= split_data.max() / 255.0

        return np.reshape(split_data, (num_frames, -1))[:-int(len(self.__noise_dummy) /
                                                              len(aud_msg_array[0])) - 1]

    #############
    # Send Data #
    #############

    def formatOutput(self, name="", start_frame=None, end_frame=None, debug=False, id=0):
        if not debug:
            self.reset()
        # Execute pre-processing on all stored input
        # if self.__use_opt_obs:
        img_stack, opt_stack = self.formatImgBatch(
            self.__img_complete_stack[start_frame:end_frame], name)
        #self.__imgStack = np.expand_dims(img_stack, axis=0)
        self.__pntStack = np.reshape(np.expand_dims(opt_stack, axis=0), (1, -1, 4096))
        self.__specStack = np.expand_dims(self.formatAudBatch(
            self.__aud_complete_stack[start_frame:end_frame], name), axis=0)
        if debug:
            cv2.imwrite('debug/audio/{}.jpg'.format(id), self.print_spectrogram(self.__specStack,
                                                                                aud_dtype))
            cv2.imwrite('debug/video/{}.jpg'.format(id), self.print_spectrogram(self.__pntStack,
                                                                                pnt_dtype))
        return self.__specStack, self.__pntStack


    def calculate_relationship(self, a_s, a_e, b_s, b_e):
        temp_distance = (np.sign(b_s - a_s), np.sign(b_e - a_e),
                         np.sign(b_s - a_e), np.sign(b_e - a_s))
        return self.__itbn_event_interval_rel_map.get(temp_distance, 0)

    def save_partial_input(self):
        if not type(self.__audStack) == int:
            if type(self.__aud_complete_stack) == int:
                self.__aud_complete_stack = self.__audStack
                self.__img_complete_stack = self.__imgStack
            else:
                self.__aud_complete_stack += self.__audStack
                self.__img_complete_stack += self.__imgStack

    def print_spectrogram(self, data, d_type):
        tout = []
        out = []
        data = data[0]
        for i in range(data.shape[0]):
            imf = np.reshape(data[i], (d_type["cmp_h"], d_type["cmp_w"], d_type["num_c"]))

            limit_size = d_type["cmp_w"]
            frame_limit = 12
            if d_type["name"] == "aud":
                frame_limit = 120

            if (d_type["cmp_w"] > limit_size):
                mod = limit_size / float(d_type["cmp_h"])
                imf = cv2.resize(imf, None, fx=mod, fy=mod, interpolation=cv2.INTER_CUBIC)

            if (imf.shape[2] == 2):
                imf = np.concatenate((imf, np.zeros((d_type["cmp_h"], d_type["cmp_w"], 1))),
                                     axis=2)
                imf[..., 0] = imf[..., 1]
                imf[..., 2] = imf[..., 1]
                imf = imf.astype(np.uint8)

            if (i % frame_limit == 0 and i != 0):
                if (len(tout) == 0):
                    tout = out.copy()
                else:
                    tout = np.concatenate((tout, out), axis=0)
                out = []
            if (len(out) == 0):
                out = imf
            else:
                out = np.concatenate((out, imf), axis=1)
        if (data.shape[0] % frame_limit != 0):
            fill = np.zeros((d_type["cmp_h"], d_type["cmp_w"] * (frame_limit -
                                                                 (data.shape[0] % frame_limit)),
                             d_type["num_c"]))  # .fill(255)
            fill.fill(0)
            out = np.concatenate((out, fill), axis=1)
        if (len(out) != 0):
            if (len(tout) == 0):
                tout = out.copy()
            else:
                tout = np.concatenate((tout, out), axis=0)
            return tout

    def get_frame_count(self):
        self.__lock.acquire()
        count = self.frame_counter
        self.__lock.release()
        return count


if __name__ == '__main__':
    packager = DQNPackager()
