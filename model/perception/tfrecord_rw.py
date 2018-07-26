#!/usr/bin/env python

# Madison Clark-Turner
# 12/2/2017
#
# Estuardo Carpio Mazariegos
# 5/30/2018

import tensorflow as tf
import numpy as np


def make_sequence_example(top_img_input, top_img_data,
                          top_grs_input, top_grs_data,
                          top_opt_input, top_opt_data,
                          nao_aud_input, nao_aud_data,
                          kinect_aud_input, kinect_aud_data,
                          temporal_info, sequence_id):
    """
    generates the tfrecord sequence

    :param top_img_input: array containing top camera image input
    :param top_img_data: dict containing the top camera image data
    :param top_grs_input: array containing top camera grays cale input
    :param top_grs_data: dict containing the top camera gray scale data
    :param top_opt_input: array containing top camera optical flow input
    :param top_opt_data: dict containing the top camera optical flow data
    :param nao_aud_input: array containing nao robot audio input
    :param nao_aud_data: dict containing the nao robot audio data
    :param kinect_aud_input: array containing kinect audio input
    :param kinect_aud_data: dict containing the kinect audio data
    :param temporal_info: dict containing the temporal information of the sequence
    :param sequence_id: string id of the sequence

    :return: a tensorflow sequence example
    """
    ex = tf.train.SequenceExample()

    sequence_length = top_opt_input.shape[1]

    ex.context.feature["length"].int64_list.value.append(sequence_length)

    ex.context.feature["top_img_h"].int64_list.value.append(top_img_data["cmp_h"])
    ex.context.feature["top_img_w"].int64_list.value.append(top_img_data["cmp_w"])
    ex.context.feature["top_img_c"].int64_list.value.append(top_img_data["num_c"])

    ex.context.feature["top_grs_h"].int64_list.value.append(top_grs_data["cmp_h"])
    ex.context.feature["top_grs_w"].int64_list.value.append(top_grs_data["cmp_w"])
    ex.context.feature["top_grs_c"].int64_list.value.append(top_grs_data["num_c"])

    ex.context.feature["top_opt_h"].int64_list.value.append(top_opt_data["cmp_h"])
    ex.context.feature["top_opt_w"].int64_list.value.append(top_opt_data["cmp_w"])
    ex.context.feature["top_opt_c"].int64_list.value.append(top_opt_data["num_c"])

    ex.context.feature["nao_aud_h"].int64_list.value.append(nao_aud_data["cmp_h"])
    ex.context.feature["nao_aud_w"].int64_list.value.append(nao_aud_data["cmp_w"])
    ex.context.feature["nao_aud_c"].int64_list.value.append(nao_aud_data["num_c"])

    ex.context.feature["kinect_aud_h"].int64_list.value.append(kinect_aud_data["cmp_h"])
    ex.context.feature["kinect_aud_w"].int64_list.value.append(kinect_aud_data["cmp_w"])
    ex.context.feature["kinect_aud_c"].int64_list.value.append(kinect_aud_data["num_c"])

    ex.context.feature["example_id"].bytes_list.value.append(sequence_id)

    timing_labels, timing_values = "", []
    for event, temp_info in temporal_info.iteritems():
        timing_labels += event + "/"
        timing_values.append(temp_info)

    ex.context.feature["temporal_labels"].bytes_list.value.append(timing_labels)

    # Feature lists for input data
    def load_array(example, name, data, dtype):
        fl_data = example.feature_lists.feature_list[name].feature.add().bytes_list.value
        fl_data.append(np.asarray(data).astype(dtype).tostring())

    load_array(ex, "top_img", top_img_input, np.uint8)
    load_array(ex, "top_grs", top_grs_input, np.uint8)
    load_array(ex, "top_opt", top_opt_input, np.uint8)
    load_array(ex, "nao_aud", nao_aud_input, np.uint8)
    load_array(ex, "kinect_aud", kinect_aud_input, np.uint8)
    load_array(ex, "temporal_values", timing_values, np.int16)

    return ex


# READ
def parse_sequence_example(input_file):
    """
    reads a TFRecord into its constituent parts

    :param input_file: file from which the tfrecord will be read

    :return: sequence context and contents
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(input_file)

    context_features = {
        "length": tf.FixedLenFeature([], dtype=tf.int64),

        "top_img_h": tf.FixedLenFeature([], dtype=tf.int64),
        "top_img_w": tf.FixedLenFeature([], dtype=tf.int64),
        "top_img_c": tf.FixedLenFeature([], dtype=tf.int64),

        "top_grs_h": tf.FixedLenFeature([], dtype=tf.int64),
        "top_grs_w": tf.FixedLenFeature([], dtype=tf.int64),
        "top_grs_c": tf.FixedLenFeature([], dtype=tf.int64),

        "top_opt_h": tf.FixedLenFeature([], dtype=tf.int64),
        "top_opt_w": tf.FixedLenFeature([], dtype=tf.int64),
        "top_opt_c": tf.FixedLenFeature([], dtype=tf.int64),

        "nao_aud_h": tf.FixedLenFeature([], dtype=tf.int64),
        "nao_aud_w": tf.FixedLenFeature([], dtype=tf.int64),
        "nao_aud_c": tf.FixedLenFeature([], dtype=tf.int64),

        "kinect_aud_h": tf.FixedLenFeature([], dtype=tf.int64),
        "kinect_aud_w": tf.FixedLenFeature([], dtype=tf.int64),
        "kinect_aud_c": tf.FixedLenFeature([], dtype=tf.int64),

        "example_id": tf.FixedLenFeature([], dtype=tf.string),
        "temporal_labels": tf.FixedLenFeature([], dtype=tf.string)
    }

    sequence_features = {
        "top_img": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "top_grs": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "top_opt": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "nao_aud": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "kinect_aud": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "temporal_values": tf.FixedLenSequenceFeature([], dtype=tf.string)
    }

    # Parse the example
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    sequence_data = {
        "top_img": tf.decode_raw(sequence_parsed["top_img"], tf.uint8),
        "top_grs": tf.decode_raw(sequence_parsed["top_grs"], tf.uint8),
        "top_opt": tf.decode_raw(sequence_parsed["top_opt"], tf.uint8),
        "nao_aud": tf.decode_raw(sequence_parsed["nao_aud"], tf.uint8),
        "kinect_aud": tf.decode_raw(sequence_parsed["kinect_aud"], tf.uint8),
        "temporal_values": tf.decode_raw(sequence_parsed["temporal_values"], tf.int16)
    }

    return context_parsed, sequence_data


def parse_timing_dict(labels, values):
    return dict(zip(labels.split('/'), values[0]))
