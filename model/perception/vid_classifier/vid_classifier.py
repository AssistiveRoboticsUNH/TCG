import tensorflow as tf

# contains information relating to input data size
from model.perception.constants import *

# network layer information
layer_elements = [-1, 16, 32, 128, len(CLASSES_V)]
output_sizes = [32, 16, 4]
filter_sizes = [4, 4, 8]
stride_sizes = [2, 2, 4]
padding_size = [1, 1, 2]


class VideoCNNClassifier:
    """
    CNN classifier for video inputs.
    Classes are defined in the perception.constants file as CLASSES_V.
    """
    def __init__(self, filename="", learning_rate=1e-5):
        """
        Constructor
        :param filename: string, location of file with saved model parameters. If empty a new model
        is trained from scratch
        :param learning_rate: - float, speed at which the model trains.
        """
        self.__batch_size = 1
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._alpha = learning_rate

            # Model variables
            def weight_variable(name, shape):
                """
                Initializes a weight variable with a normal distribution and 0.1 standard deviation
                :param name: variable name
                :param shape: variable shape
                :return: initialized tf.Variable object
                """
                initial = tf.truncated_normal(shape, stddev=0.1)
                return tf.Variable(initial, name=name)

            def bias_variable(name, shape):
                """
                Initializes a bias variable with a 0.1 value for every cell
                :param name: variable name
                :param shape: variable shape
                :return: initialized tf.Variable
                """
                initial = tf.constant(0.1, shape=shape)
                return tf.Variable(initial, name=name)

            self.variables_opt = {
                "W1": weight_variable("W_conv1_pnt", [filter_sizes[0], filter_sizes[0],
                                                      opt_dtype["num_c"], layer_elements[1]]),
                "b1": bias_variable("b_conv1_pnt", [layer_elements[1]]),
                "W2": weight_variable("W_conv2_pnt", [filter_sizes[1], filter_sizes[1],
                                                      layer_elements[1], layer_elements[2]]),
                "b2": bias_variable("b_conv2_pnt", [layer_elements[2]]),
                "W3": weight_variable("W_conv3_pnt", [filter_sizes[2], filter_sizes[2],
                                                      layer_elements[2], layer_elements[-2]]),
                "b3": bias_variable("b_conv3_pnt", [layer_elements[-2]]),
                "W_lstm": weight_variable("W_lstm", [layer_elements[-2], layer_elements[-1]]),
                "b_lstm": bias_variable("b_lstm", [layer_elements[-1]]),
                "W_fc": weight_variable("W_fc", [layer_elements[-1] + 1, layer_elements[-1]]),
                "b_fc": bias_variable("b_fc", [layer_elements[-1]])
            }

            # Placeholder variables
            # placeholder for the Optical Flow data
            self.opt_ph = tf.placeholder("float",
                                         [self.__batch_size, None, grs_dtype["cmp_h"] *
                                          grs_dtype["cmp_w"] * grs_dtype["num_c"]],
                                         name="opt_placeholder")

            # placeholder for the sequence length
            self.seq_length_ph = tf.placeholder("int32", [self.__batch_size],
                                                name="seq_len_placeholder")

            # placeholder for the correct classification of a sequence
            self.opt_y_ph = tf.placeholder("float", [None, len(CLASSES_V)],
                                           name="pnt_y_placeholder")

            # Build Model Structure
            # initialize all variables in the network
            self.pred_opt_set = self.execute_opt_var_set()

            # returns the classification for the given sequence
            self.opt_observed = tf.argmax(self.execute_opt(), 1)
            self.observe = tf.argmax(self.execute_opt(), 1)

            # Optimization Functions
            self.cross_entropy_opt = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.opt_y_ph, logits=self.execute_opt())

            # optimize the network
            self.optimizer_opt = tf.train.AdamOptimizer(learning_rate=self._alpha).minimize(
                self.cross_entropy_opt)

            # Evaluation Functions
            # return a boolean indicating whether the system correctly predicted the output
            self.correct_pred_opt = tf.equal(tf.argmax(self.opt_observed, 1),
                                             tf.argmax(self.opt_y_ph, 1))

            # the accuracy of the current batch
            self.accuracy_opt = tf.reduce_mean(tf.cast(self.correct_pred_opt, tf.float32))

        # Initialization
        # Generate Session
        self.sess = tf.InteractiveSession(graph=self.graph,
                                          config=tf.ConfigProto(allow_soft_placement=True))

        # Variable for generating a save checkpoint
        self.saver = tf.train.Saver()

        if len(filename) == 0:
            # initialize all model variables
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)
            print("VARIABLE VALUES INITIALIZED")
        else:
            # restore variables from a checkpoint
            self.saver.restore(self.sess, filename)
            print("VARIABLE VALUES RESTORED FROM: " + filename)

    def save_model(self, name="model.ckpt", save_dir=""):
        """
            Saves the model to a checkpoint file
            :param name: (string) name of the checkpoint file
            :param save_dir: (string) directory to save the file into
        """
        self.saver.save(self.sess, save_dir + '/' + name)

    def execute_opt_var_set(self):
        """
        Initializes model structure
        :return: Initialized aud cnn model
        """
        return self.opt_model(
            self.seq_length_ph,
            self.opt_ph,
            tf.variable_scope("opt"),
            tf.variable_scope("opt"),
            self.variables_opt
        )

    def execute_opt(self):
        """Classifies a given input sequence"""
        return self.opt_model(
            self.seq_length_ph,
            self.opt_ph,
            tf.variable_scope("opt"),
            tf.variable_scope("opt", reuse=True),
            self.variables_opt
        )

    def classify_input_ros(self, num_frames, opt_data, verbose=False):
        """
        Classify an input passed in as separate data points. Used ROS
        to run the model without having to import tensorflow
        :param num_frames: (int) the number of frames in the video
        :param opt_data: (numpy array) an array that contains the optical flow data
        :param verbose: (bool) print additional information
        """
        opt_pred = self.sess.run(self.observe, feed_dict={
            self.seq_length_ph: [num_frames],
            self.opt_ph: opt_data
        })

        if verbose:
            print("Classification: " + CLASSES_V[int(opt_pred[0])])

        return int(opt_pred[0])

    def process_vars(self, seq, data_type):
        """
        Reshapes variables
        :param seq: input sequence
        :param data_type: input data type
        :return: reshaped input variables
        """
        seq_inp = tf.cast(seq, tf.float32)
        return tf.reshape(seq_inp, (self.__batch_size, -1, data_type["cmp_h"],
                                    data_type["cmp_w"], data_type["num_c"]))

    @staticmethod
    def check_legal_inputs(tensor, name):
        """
        Verifies that the given input is valid
        :param tensor: input tensor
        :param name: tensor name
        :return: Error message if tensor contains invalid values
        """
        return tf.verify_tensor_all_finite(tensor, "ERR: Tensor not finite - " + name, name=name)

    def opt_model(self, seq_length, pnt_ph, variable_scope, variable_scope2, var_opt):
        """
        CNN model structure
        :param seq_length: (placeholder) the number of frames in the input
        :param opt_ph: (placeholder) an array that contains the optical flow data
        :param variable_scope: (variable_scope) scope for the optical flow data
        :param variable_scope2: (variable_scope) scope for the temporal data
        :param var_aud: (dict) the variables for the audio input
        """
        # Convolution Functions
        def convolve_data_3layer_pnt(input_data, variables, n, dtype):
            # pass data into through V_CNN
            def pad_tf(x, p):
                return tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "CONSTANT")

            def gen_convolved_output(sequence, W, b, stride, num_hidden, new_size, padding='SAME'):
                conv = tf.nn.conv2d(sequence, W, strides=[1, stride, stride, 1],
                                    padding=padding) + b
                return tf.nn.relu(conv)

            input_data = tf.reshape(input_data,
                                    [-1, dtype["cmp_h"], dtype["cmp_w"], dtype["num_c"]],
                                    name=n + "_inp_reshape")

            for i in range(3):
                si = str(i + 1)

                input_data = pad_tf(input_data, padding_size[i])
                padding = "VALID"

                input_data = gen_convolved_output(input_data, variables["W" + si],
                                                  variables["b" + si], stride_sizes[i],
                                                  layer_elements[i + 1], output_sizes[i], padding)
                input_data = self.check_legal_inputs(input_data, "conv" + si + "_" + n)

            return input_data

        # Model Execution Begins Here
        # CNN Stacks
        # Inception Network (INRV2)
        with tf.device(VID_GPU):
            with variable_scope as scope:
                # P_CNN
                inp_data = self.process_vars(pnt_ph, grs_dtype)
                conv_inp = convolve_data_3layer_pnt(inp_data, var_opt, "opt", grs_dtype)
                conv_inp = tf.reshape(conv_inp, [self.__batch_size, -1,
                                                 output_sizes[-1] * output_sizes[-1] *
                                                 layer_elements[-2]], name="combine_reshape")

                # capture variables before changing scope
                W_lstm = var_opt["W_lstm"]
                b_lstm = var_opt["b_lstm"]

        with variable_scope2 as scope:
            # Internal Temporal Information (LSTM)
            lstm_cell = tf.contrib.rnn.LSTMCell(layer_elements[-2],
                                                use_peepholes=False,
                                                cell_clip=None,
                                                initializer=None,
                                                num_proj=None,
                                                proj_clip=None,
                                                forget_bias=1.0,
                                                state_is_tuple=True,
                                                activation=None,
                                                reuse=None
                                                )

            lstm_mat, _ = tf.nn.dynamic_rnn(
                cell=lstm_cell,
                inputs=conv_inp,
                dtype=tf.float32,
                sequence_length=seq_length,
                time_major=False
            )

            # if lstm_out is NaN replace with 0 to prevent model breakage
            lstm_mat = tf.where(tf.is_nan(lstm_mat), tf.zeros_like(lstm_mat), lstm_mat)
            lstm_mat = self.check_legal_inputs(lstm_mat, "lstm_mat")

            # extract relevant information from LSTM output using partitions
            lstm_out = tf.expand_dims(lstm_mat[0, -1], 0)

            # FC1
            fc1_out = tf.matmul(lstm_out, W_lstm) + b_lstm
            fc1_out = self.check_legal_inputs(fc1_out, "fc1")

            return fc1_out


if __name__ == '__main__':
    cnn = VideoCNNClassifier()
