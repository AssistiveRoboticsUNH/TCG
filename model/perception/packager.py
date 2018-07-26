# Madison Clark-Turner
# 10/14/2017
#
# Estuardo Carpio
# 5/30/2018

import os
import math
import threading
import numpy as np

# image pre-processing and optical flow generation
import cv2
from cv_bridge import CvBridge

# audio pre-processing
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy import signal
from noise_subtraction import reduce_noise

from constants import *


class CNNPackager:
    """
    DQN Packager listens to the topics for images and audio.
    Processes those inputs into sequences and passes the result to the CNN model.
    """
    def __init__(self, flip=False, generate_samples=False, input_dir=INPUT_DIRECTORY):
        """
        Class constructor
        :param flip: boolean indicating if the input image feeds need to be flipped horizontally
        """
        self._flip = flip
        self._generate_samples = generate_samples
        self._input_dir = input_dir

        # Lock to protect shared arrays
        self._lock = threading.Lock()

        # Variables for tracking captured data from image and audio feeds
        self._img_stack = []
        self._grs_stack = []
        self._opt_stack = []
        self._nao_aud_stack = []
        self._kinect_aud_stack = []
        self._recent_msgs = [list(), list(), list()]
        self.nao_spectrogram = None
        self.kinect_spectrogram = None
        try:
            self._nao_noise_sample = np.load(os.path.join(self._input_dir, 'nao_noise.npy'))
            self._kinect_noise_sample = np.load(os.path.join(self._input_dir, 'kinect_noise.npy'))
        except IOError:
            self._nao_noise_sample = None
            self._kinect_noise_sample = None
            print('Warning: Noise samples not found.')
        self._nao_rate = 16000
        self._face_cascade = cv2.CascadeClassifier(os.path.join(
            input_dir, 'haarcascade_frontalface_default.xml'))

        # Variables for optical flow
        self._previous_frame = None
        self._max_opt_frame = None
        self._max_opt_mean = 0
        self._nao_spect_sample = None
        self._max_nao_spect_mean = 0
        self._kinect_spect_sample = None
        self._max_kinect_spect_mean = 0
        if not self._generate_samples:
            try:
                self._nao_spect_sample = np.load(os.path.join(
                    self._input_dir, 'nao_spect_sample.npy'))
                self._kinect_spect_sample = np.load(os.path.join(
                    self._input_dir, 'kinect_spect_sample.npy'))
                self._opt_sample = np.load(os.path.join(self._input_dir, 'opt_sample.npy'))
            except IOError:
                print('Warning: Visual samples not found.')

    def get_img_stack(self):
        """
        :return: Stack containing the captured image data
        """
        return self._img_stack

    def get_grs_stack(self):
        """
        :return: Stack containing the generated gray scale image data
        """
        return self._grs_stack

    def get_pnt_stack(self):
        """
        :return: Stack containing the generated optical flow data
        """
        return self._opt_stack

    def get_nao_aud_stack(self):
        """
        :return: Stack containing the captured nao audio data
        """
        return self._nao_aud_stack

    def get_kinect_aud_stack(self):
        """
        :return: Stack containing the generated kinect audio data
        """
        return self._kinect_aud_stack

    def get_img_frame_count(self):
        """
        :return: size of captured image data
        """
        return len(self._img_stack)

    def get_kinect_frame_count(self):
        """
        :return: size of capture kinect audio
        """
        return len(self._kinect_aud_stack)

    def reset(self, already_locked=False):
        """
        Resets the packager and its attributes to their initial state.
        This procedure is thread safe.
        :param already_locked: boolean indicating if the lock has already been acquired.
        """
        if not already_locked:
            self._lock.acquire()

        self._img_stack = []
        self._grs_stack = []
        self._opt_stack = []
        self._nao_aud_stack = []
        self._kinect_aud_stack = []
        self._recent_msgs = [list(), list(), list()]

        # Variables for optical flow
        self._previous_frame = None

        if not already_locked:
            self._lock.release()

    def img_callback(self, msg):
        """
        Method executed when a message from the image topic is received
        :param msg: received message. Type: sensor_msgs.msg.Image
        """
        self._recent_msgs[0].append(msg)
        self.check_msgs()

    def nao_aud_callback(self, msg):
        """
        Method executed when a message from the nao audio topic is received
        :param msg: received message. Type: nao_msgs.msg.AudioBuffer
        """
        self._recent_msgs[1].append(self.format_nao_aud_msg(msg))
        self.check_msgs()

    def kinect_aud_callback(self, msg):
        """
        Method executed when a message from the kinect audio topic is received
        :param msg: received message. Type: audio_common_msgs.msg.AudioData
        """
        self._recent_msgs[2].append(msg)
        self.check_msgs()

    @staticmethod
    def format_nao_aud_msg(aud_msg):
        """
        Shapes the nao audio messages for later use.
        :param aud_msg: nao audio message
        :return: formatted audio data
        """
        data = aud_msg.data
        data = np.reshape(data, (-1, 4))
        data = data.transpose([1, 0])
        return data[0]

    def check_msgs(self):
        """
        Stores received messages into the final stacks
        """
        self._lock.acquire()

        self._img_stack += self._recent_msgs[0]
        self._nao_aud_stack += self._recent_msgs[1]
        self._kinect_aud_stack += self._recent_msgs[2]

        self._recent_msgs = [list(), list(), list()]
        self._lock.release()

    # FORMAT DATA
    def format_img_batch(self, img_stack):
        """
        Pre-process image stack and generates gray scale and optical flow stacks
        :param img_stack: Accumulated image input
        :return: formatted image, gray scale and optical flow stacks
        """
        img_out, grs_out, opt_out = [], [], []

        if type(img_stack) != int:
            for x in img_stack:
                img, grs = self.format_img(x)
                img_out.append(np.asarray(img).flatten())
                grs_out.append(np.asarray(grs).flatten())
                opt_out.append(self.format_opt(grs))
            # reset self._previous_frame
            self._previous_frame = None

        return img_out, grs_out, opt_out

    def format_img(self, img_msg):
        """
        Formats an image message and generates its gray scale equivalent
        :param img_msg: image message to be formatted
        :return: formatted image message and its gray scale equivalent
        """
        # pre-process the image data to crop it to an appropriate size
        # convert image to cv2 image and generate gray scale
        img = CvBridge().imgmsg_to_cv2(img_msg, "bgr8")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # identify location of face if possible using Haar cascade and crop to center on it.
        faces = self._face_cascade.detectMultiScale(gray, 1.3, 5)

        buff = img.shape[1] - img.shape[0]
        x, y, w, h = -1, -1, -1, -1

        if len(faces) > 0:
            # if face is located to the edge set then crop from the opposite side
            for (xf, yf, wf, hf) in faces:
                if wf * hf > w * h:
                    x, y, w, h = xf, yf, wf, hf

            if x >= 0 and y >= 0 and x + w < (img_w / 2) and y + h < (img_h / 2):
                y, h = 0, img.shape[0]
                mid = x + (w / 2)
                x, w = mid - (img.shape[0] / 2), img.shape[0]
                if x < 0:
                    x = 0
                elif x > buff:
                    x = buff / 2
                img = img[y: y + h, x: x + w]
        else:
            # if no face visible set crop image to center of the video
            diff = img.shape[1] - img.shape[0]
            img = img[0:img.shape[0], (diff / 2):(img.shape[1] - diff / 2)]

        # resize image to 299 x 299
        y_mod = 1 / (img.shape[0] / float(img_dtype["cmp_h"]))
        x_mod = 1 / (img.shape[1] / float(img_dtype["cmp_w"]))
        img = cv2.resize(img, None, fx=x_mod, fy=y_mod, interpolation=cv2.INTER_CUBIC)

        if self._flip:
            # if flip set to true then mirror the image horizontally
            img = np.flip(img, 1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img, gray

    def format_opt(self, img_src):
        """
        Generates the optical flow given an image input
        :param img_src: gray scale image input that will be used to generate the optical flow
        :return: optical flow for the given image
        """
        # generate optical flow
        mod = opt_dtype["cmp_h"] / float(img_dtype["cmp_h"])
        img = cv2.resize(img_src.copy(), None, fx=mod, fy=mod, interpolation=cv2.INTER_CUBIC)
        opt_img = np.zeros(img.shape)

        if self._previous_frame is None:
            self._previous_frame = img
        else:
            # generate optical flow
            new_frame = img
            flow = cv2.calcOpticalFlowFarneback(self._previous_frame, new_frame,
                                                None, 0.5, 1, 2, 5, 7, 1.5, 1)
            opt_img, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # normalize the magnitude to between 0 and 255
            if self._generate_samples:
                mean = np.mean(opt_img)
                if mean > self._max_opt_mean:
                    self._max_opt_mean = mean
                    self._max_opt_frame = opt_img
            else:
                opt_img = np.append(opt_img, self._opt_sample, 0)
            opt_img = cv2.normalize(opt_img, None, 0, 255, cv2.NORM_MINMAX)
            self._previous_frame = new_frame
            if not self._generate_samples:
                opt_img = opt_img[:-len(self._opt_sample)]

        return np.asarray(opt_img).flatten()

    def format_nao_aud_batch(self, aud_msg_array):
        """
        Formats the received nao audio feed
        :param aud_msg_array: Accumulated audio input
        :return: formatted mel-spectrogram equivalent to the audio input
        """
        num_frames = len(aud_msg_array)
        input_data = np.reshape(aud_msg_array, (num_frames * len(aud_msg_array[0])))
        return self.generate_spectrogram(input_data, 'nao')

    def format_kinect_aud_batch(self, aud_msg_array):
        """
        Formats the received kinect audio feed
        :param aud_msg_array: Accumulated audio input
        :return: formatted mel-spectrogram equivalent to the audio input
        """
        tmp_path = 'tmp_file.mp3'
        with open(tmp_path, 'w') as tmp_file:
            for msg in aud_msg_array:
                tmp_file.write(msg.data)
        input_data, rate = librosa.load(tmp_path, sr=self._nao_rate)
        os.remove(tmp_path)
        return self.generate_spectrogram(input_data, 'kinect')

    def generate_spectrogram(self, audio_data, source):
        """
        Generates a mel-spectrogram for the given audio data
        :param audio_data: ndarray that will be processed
        :param source: string indicating if the audio source is a 'kinect' sensor or 'nao' robot
        :return: generated mel-spectrogram
        """
        num_frames = int(float(len(audio_data)) / self._nao_rate * 10)
        if 'kinect' in source:
            noise = self._kinect_noise_sample[:len(audio_data)]
            spect_sample = self._kinect_spect_sample
            max_mean = self._max_kinect_spect_mean
        else:
            noise = self._nao_noise_sample
            spect_sample = self._nao_spect_sample
            max_mean = self._max_nao_spect_mean

        if noise is None:
            noise_sample_s, noise_sample_e = int(self._nao_rate * (-1.5)), -1
            noise = audio_data[noise_sample_s, noise_sample_e]

        # perform spectral subtraction to reduce noise
        filtered_input = reduce_noise(np.array(audio_data), noise)

        # smooth signal
        b, a = signal.butter(3, [0.05])
        filtered_input = signal.lfilter(b, a, filtered_input)
        # noise = filtered_input[noise_sample_s: noise_sample_e]

        # additional spectral subtraction to remove remaining noise
        filtered_input = reduce_noise(filtered_input, noise)

        # attach spectrogram sample
        if self._generate_samples:
            frame_size = len(filtered_input) / num_frames
            for i in range(num_frames):
                curr_start = i * frame_size
                mean = np.mean(filtered_input[curr_start:curr_start + frame_size])
                if mean > max_mean:
                    max_mean = mean
                    spect_sample = filtered_input[curr_start:curr_start + frame_size]
        else:
            filtered_input = np.append(filtered_input, spect_sample)
            num_frames += 1

        # generate spectrogram
        spectrogram = librosa.feature.melspectrogram(y=filtered_input, sr=self._nao_rate,
                                                     n_mels=128, fmax=8000)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        if 'kinect' in source:
            self.kinect_spectrogram = spectrogram
        else:
            self.nao_spectrogram = spectrogram

        # split the spectrogram into A_i. This generates an overlap between
        # frames with as set stride
        stride = spectrogram.shape[1] / float(num_frames)
        frame_len = aud_dtype["cmp_w"]

        # pad the entire spectrogram so that overlaps at either end do not fall out of bounds
        min_val = np.nanmin(spectrogram)
        empty = np.zeros((spectrogram.shape[0], 8))
        empty.fill(min_val)
        empty_end = np.zeros((spectrogram.shape[0], 8))
        empty_end.fill(min_val)
        spectrogram = np.concatenate((empty, spectrogram, empty_end), axis=1)

        split_data = np.zeros(shape=(num_frames, spectrogram.shape[0], frame_len),
                              dtype=spectrogram.dtype)
        for i in range(0, num_frames):
            split_data[i] = spectrogram[:, int(math.floor(i * stride)):
            int(math.floor(i * stride)) + frame_len]

        # normalize the output to be between 0 and 255
        split_data -= split_data.min()
        split_data /= split_data.max() / 255.0

        if not self._generate_samples:
            split_data = split_data[:-1]
            num_frames -= 1
        else:
            if 'kinect' in source:
                self._kinect_spect_sample = spect_sample
                self._max_kinect_spect_mean = max_mean
            else:
                self._nao_spect_sample = spect_sample
                self._max_nao_spect_mean = max_mean

        return np.reshape(split_data, (num_frames, -1))

    def format_output(self, debug=False, output_path=""):
        """
        Execute pre-processing on all stored input
        :param debug: boolean indicating if debugging output will be generated
        :param output_path: directory in which debugging output will be generated
        """
        top_img_stack, top_grs_stack, top_opt_stack = self.format_img_batch(self._img_stack)
        self._img_stack = np.expand_dims(top_img_stack, axis=0)
        self._grs_stack = np.expand_dims(top_grs_stack, axis=0)
        self._opt_stack = np.expand_dims(top_opt_stack, axis=0)
        self._nao_aud_stack = np.expand_dims(self.format_nao_aud_batch(self._nao_aud_stack), axis=0)
        self._kinect_aud_stack = np.expand_dims(self.format_kinect_aud_batch(
            self._kinect_aud_stack), axis=0)
        if debug:
            # self.output_spectrograms(output_path)
            pass

    def format_kinect_range(self, start, end):
        """
        Formats a given range of the received kinect audio
        :param start: int indicating the start frame of the range
        :param end: int indicating the last frame of the range
        :return: array containing the desired range in a mel-spectrogram format
        """
        return np.expand_dims(self.format_kinect_aud_batch(self._kinect_aud_stack[start: end]),
                              axis=0)

    def output_spectrograms(self, output_path):
        """
        Generates the spectrograms for the nao and kinect audio inputs
        :param output_path: path in which the files will be stored
        """
        plt.figure(figsize=(12, 4))
        if self.nao_spectrogram is not None:
            librosa.display.specshow(self.nao_spectrogram)
            plt.savefig(output_path + '_nspect.png', bbox_inches='tight', pad_inches=0)
            plt.clf()
        if self.kinect_spectrogram is not None:
            librosa.display.specshow(self.kinect_spectrogram)
            plt.savefig(output_path + '_kspect.png', bbox_inches='tight', pad_inches=0)
            plt.clf()

    def nao_noise_callback(self, msg):
        """
        Callback used to collect the audio containing a noise sample
        :param msg: audio message received from the nao robot
        """
        self._nao_aud_stack.append(self.format_nao_aud_msg(msg))

    def kinect_noise_callback(self, msg):
        """
        Callback used to collect the audio containing a noise sample
        :param msg: audio message received from the kinect
        """
        self._kinect_aud_stack.append(msg)

    def save_noise_sample(self):
        """
        Method that starts the process of saving a noise sample from the nao and kinect feeds
        """
        self.save_nao_noise()
        self.save_kinect_noise()

    def save_nao_noise(self):
        """
        Stores the noise sample for the nao audio input source
        """
        num_frames = len(self._nao_aud_stack)
        noise_data = np.reshape(self._nao_aud_stack, (num_frames * len(self._nao_aud_stack[0])))
        np.save(os.path.join(self._input_dir, 'nao_noise.npy'), noise_data)

    def save_kinect_noise(self):
        """
        Saves the audio input for the kinect audio input source
        """
        tmp_path = 'tmp_file.mp3'
        tmp_file = open(tmp_path, 'w')
        for msg in self._kinect_aud_stack:
            tmp_file.write(msg.data)
        tmp_file.close()
        noise_data, rate = librosa.load(tmp_path)
        os.remove(tmp_path)
        np.save(os.path.join(self._input_dir, 'kinect_noise.npy'), noise_data)

    def save_visual_samples(self):
        """
        Saves image samples to disk.
        """
        np.save(os.path.join(self._input_dir + 'opt_sample.npy'), self._max_opt_frame)
        np.save(os.path.join(self._input_dir + 'nao_spect_sample.npy'), self._nao_spect_sample)
        np.save(os.path.join(self._input_dir + 'kinect_spect_sample.npy'),
                self._kinect_spect_sample)

    def save_kinect_input(self, limit):
        """
        Saves the audio input for the kinect audio input source in an mp3 file
        """
        tmp_path = '_tmp_file.mp3'
        tmp_file = open(tmp_path, 'w')
        for msg in self._kinect_aud_stack[0:limit]:
            tmp_file.write(msg.data)
        tmp_file.close()


if __name__ == '__main__':
    packager = CNNPackager()
