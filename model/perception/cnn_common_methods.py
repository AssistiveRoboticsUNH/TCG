# Script containing methods used in the aud and vid CNNs
import os

from model.perception.constants import *
from model.TCG.IntervalAlgebra import *


class CNNHelper:
    VALID_ITR_SET = ['during', 'during_i',
                     'overlaps', 'overlaps_i',
                     'starts', 'starts_i',
                     'finishes', 'finishes_i',
                     'equal']
    START_SUFFIX = '_s'
    END_SUFFIX = '_e'
    FIELDS_SEPARATOR = ' '
    FIELD_NAME_SEPARATOR = '_'

    def __init__(self, cnn_classes, default_class, dataset_base_dir, validation_file_path,
                 tfrecords_dir, window_size, window_stride):
        """
        Class constructor
        :param cnn_classes: list containing the names of the classes classified by the cnn
        :param default_class: str class to be assigned to windows that don't contain any event.
        :param dataset_base_dir: str indicating the base directory for the dataset
        :param validation_file_path: str indicates the path of a .txt file containing the list of
        :param tfrecords_dir: str indicating the directory within the base directory where the
        tfrecords are stored.
        files that form the VALIDATION set.
        """
        self.cnn_classes = cnn_classes
        self.default_class = default_class
        self.default_class_index = self.cnn_classes.index(self.default_class)
        self.base_dir = dataset_base_dir
        self.tfrecords_dir = tfrecords_dir
        self.window_size = window_size
        self.window_stride = window_stride
        self.validation_set = list()
        with open(os.path.join(dataset_base_dir, validation_file_path), 'r') as val_set_file:
            for line in val_set_file.readlines():
                self.validation_set.append(
                    line[:-1].replace('bags', 'tfrecords').replace('.bag', '.tfrecord'))

    @staticmethod
    def check_window_overlap(s_time, e_time, td, label):
        """
        Determines if the a window starting at s_time and ending at e_time contains an event of the
        given label
        :param s_time: int indicating the frame number at which the window starts.
        :param e_time: int indicating the frame number at which the window ends.
        :param td: dict mapping from event names to start times.
        :param label: str indicates the name of the event of interest.
        :return: boolean indicating if the event is contained in the given window
        """
        s_label = label + CNNHelper.START_SUFFIX
        e_label = label + CNNHelper.END_SUFFIX
        window = AtomicEvent(label, s_time, e_time)
        event = AtomicEvent(label, td[s_label], td[e_label])
        if IntervalAlgebra.obtain_itr(window, event).name in CNNHelper.VALID_ITR_SET:
            return True
        return False

    def label_window(self, s_frame, e_frame, timing_dict):
        """
        Assigns a label from the given cnn classes to the window defined by the
        provided start and end frames.
        :param s_frame: int indicating the frame number at which the window starts.
        :param e_frame: int indicating the frame number at which the window ends.
        :param timing_dict: dict mapping from event names to start times.
        :return: a one-hot activated vector indicating the class of the observation
        """
        label_array = np.zeros((1, len(self.cnn_classes))).astype(float)
        observed_label = self.default_class_index
        for event in timing_dict.keys():
            if event.endswith(CNNHelper.START_SUFFIX):
                event_class = OBS_TO_CLASS.get(event.split(CNNHelper.FIELD_NAME_SEPARATOR)[0])
                if event_class in self.cnn_classes and self.check_window_overlap(
                        s_frame, e_frame, timing_dict, event[:-len(CNNHelper.START_SUFFIX)]):
                    observed_label = min(observed_label, self.cnn_classes.index(event_class))
        label_array[0][observed_label] = 1
        return label_array

    def adjust_timing_dict_for_audio(self, timing_dict, input_file, fps=10):
        """
        Modifies the timing values of each event to match the fps of the audio recording device
        :param timing_dict: dict mapping from event names to start times.
        :param input_file: name of the file containing the adjusted timing information
        :param fps: int indicating the frames per second of the audio recording device
        """
        with open(input_file, 'r') as o_file:
            lines = o_file.readlines()
        for line in lines:
            event, event_time = line.split(CNNHelper.FIELDS_SEPARATOR)
            event_name = event.split(CNNHelper.FIELD_NAME_SEPARATOR)[0]
            if OBS_TO_CLASS.get(event_name) in self.cnn_classes:
                timing_dict[event] = int(round(float(event_time) * fps))

    def get_files_set(self, validation=False):
        """
        Returns the set of files to be used by the cnn trainer or validator
        :param validation: boolean indicating if the VALIDATION set should be used
        :return: list of files to be used by the trainer or validator
        """
        if validation:
            return self.validation_set
        files_set = list()
        for root, subdir, files in os.walk(os.path.join(self.base_dir, self.tfrecords_dir)):
            for f in files:
                if '.tfrecord' in f and os.path.join(root, f) not in self.validation_set:
                    files_set.append(os.path.join(root, f))
        files_set.sort()
        return files_set

    def update_window_limits(self, window_counter, random_skew=0):
        """
        Updates the start and end positions of the current window as well as the window counter
        :param window_counter: int indicating the number of windows that have been processed
        :return: start: int indicating the start position of the window
                 end: int indicating the end position of the window
                 window_counter: int incremented value window counter
        """
        window_counter += 1
        start = self.window_stride * window_counter + random_skew
        end = start + self.window_size
        return start, end, window_counter
