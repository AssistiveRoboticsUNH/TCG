import os
import sys
import IntervalAlgebra as ia
import NGram as ng


class TCGEdge:
    """Models an edge of a Temporal Context Graph"""
    def __init__(self, start_node, end_node, event):
        """
        Constructor
        :param start_node: node where the edge origins.
        :param end_node: node where the edge ends.
        :param event: event that generates the transition.
        """
        self.start = start_node
        self.end = end_node
        self.event = event

    def __repr__(self):
        return '{} -> {} -> {}'.format(self.start, self.event, self.end)


class TCGNode:
    """Models a node of a Temporal Context Graph"""
    def __init__(self, event, is_start=False, is_terminal=False,
                 is_transition=False, duration=0, timeout=0, edges=None):
        """
        Constructor
        :param is_start: boolean indicating if the node is a start node in the network.
        :param is_terminal: boolean indicating if the node is a terminal node in the network.
        :param is_transition: boolean indicating if the can represent a transition event.
        :param duration: float indicating the duration of the event in seconds.
        :param timeout: float indicating the time in seconds before a timeout transition is
            executed. If this value is negative, a timeout transition is never executed.
        :param edges: set of edges originating from the node.
        """
        self.event = event
        self.is_start = is_start
        self.is_terminal = is_terminal
        self.is_transition = is_transition
        self.duration = duration
        self.timeout = timeout
        self.ngram_order = -1
        if edges is None:
            self.edges = dict()
        else:
            self.edges = edges

    def __repr__(self):
        out = '{} d:{:.2f} t:{:.2f} o:{}'.format(self.event, self.duration,
                                                 self.timeout, self.ngram_order)
        if self.is_start:
            out += ' (s)'
        if self.is_terminal:
            out += ' (t)'
        if self.is_transition:
            out += ' (tr)'
        return out


class TCGEvent:
    """Models an event used to learn a Temporal Context Graph"""
    def __init__(self, str_event, tcg):
        """
        Constructor.
        :param str_event: string representation of the event containing all its information
        :param tcg: Temporal Context Graph. Used to retrieve the constants
            needed to process input files.
        """
        event_info, event_time = str_event.split(tcg.info_separator)
        event_id = event_info.split(tcg.name_separator)[0]
        self.symbol = tcg.event_symbols.get(event_id, event_id)
        self.name = event_info.replace(tcg.start_marker, '').replace(tcg.end_marker, '')
        if tcg.start_marker in event_info:
            self.start = float(event_time)
            self.end = None
        else:
            self.end = float(event_time)
            self.start = None


class TemporalContextGraph:
    """Class created to model a Temporal Context Graph"""

    TIMEOUT = 'T/O'
    PERCEPTION_DELAY = 1.0

    def __init__(self, transition_events=list()):
        """
        Constructor
        :param transition_events: set of events that prompt a transition.
        """
        # Model structure and parameters
        self.edges = dict()
        self.nodes = dict()
        self.ngrams = dict()
        self.transition_events = transition_events
        self.transitions_cache = None
        self.itr_cache = None

        # Model learning
        self.start_marker = '_s'
        self.end_marker = '_e'
        self.info_separator = ' '
        self.name_separator = '_'
        self.event_symbols = None

        # Task execution and inference
        self.state = None
        self.fail_state = None
        self.observation = None
        self.sequence = None
        self.timeout = -1

    # LEARNING METHODS #############################################################################
    def learn_model_from_files(self, temporal_files_dir, start_marker='_s', end_marker='_e',
                               info_separator=' ', name_separator='_', event_symbols=None,
                               validation_file_path='validation_set.txt', ngrams=list()):
        """
        Learns a Temporal Context Graph from a set of input files.
        :param temporal_files_dir: string indicating the location of the input files.
        :param start_marker: string indicating the suffix that marks the start of an event in the
            input files.
        :param end_marker: string indicating the suffix that marks the end of an event in the
            input files.
        :param info_separator: string indicating the character that separates an event id and its
            time of occurrence in the input files.
        :param name_separator: string indicating the character that separates the fields in the
            event ids in the input files.
        :param event_symbols: dictionary mapping from event ids to event symbols.
        :param validation_file_path: string indicating the path to the file that contains the name
            of the input files that need to ignored during the training of the model.
        :param ngrams: list containing the order of n-grams to be learned. If the list is empty
            a variable length approach is used.
        """
        if event_symbols is None:
            event_symbols = dict()
        self.start_marker = start_marker
        self.end_marker = end_marker
        self.info_separator = info_separator
        self.name_separator = name_separator
        self.event_symbols = event_symbols
        sequences, itr_sequences = self.process_temporal_files(
            temporal_files_dir, os.path.join(temporal_files_dir, validation_file_path))
        self.learn_structure(sequences)
        print(sequences)
        print(itr_sequences)
        gram_orders = TemporalContextGraph.process_itr_sequences(itr_sequences)
        if len(ngrams) == 0:
            for event, order in gram_orders.iteritems():
                self.nodes[event].ngram_order = order + 1
            ngrams = set([x + 1 for x in gram_orders.values()])
        for order in ngrams:
            model = ng.NGramModel(order)
            self.ngrams[order] = model
            model.learn_model(itr_sequences)

    def learn_structure(self, sequences):
        """
        Learns a Temporal Context Graph from a set of sequences
        :param sequences: set of sequences that will be used to learn the model.
            Each sequence consists of a list of events.
        """
        for seq in sequences:
            for i, event in enumerate(seq):
                if event not in self.transition_events:
                    transitions = set()
                    next_valid_event = None
                    for next_event in seq[i + 1:]:
                        if next_event in self.transition_events:
                            transitions.add(next_event)
                        else:
                            next_valid_event = next_event
                            break
                    if len(transitions) == 0:
                        transitions.add(TemporalContextGraph.TIMEOUT)
                    if next_valid_event is not None:
                        for t in transitions:
                            edge = (event, next_valid_event, t)
                            if edge not in self.edges:
                                new_edge = TCGEdge(event, next_valid_event, t)
                                self.edges[edge] = new_edge
                                self.nodes[event].edges[edge] = new_edge

    def process_temporal_files(self, root_directory, validation_set_file_path):
        """
        Learns the nodes of the input dataset and generate the set of sequences and
        Interval Temporal Relations based sequences that will be used to learn the structure
        of the Temporal Context Graph and the associated n-grams.
        :param root_directory: string indicating the path to the root directory of the dataset.
        :param validation_set_file_path: string indicating the location of the file that contains
            the files that will be used in the VALIDATION set. These files are excluded from the
            learning phase.
        :return: set of event sequences and set of itr sequences.
        """
        sequences = list()
        itr_sequences = list()
        event_counts = dict()
        timeout_counts = dict()
        self.transition_cache = dict()
        self.itr_cache = dict()
        transition_counter = 0
        validation_set = TemporalContextGraph.load_validation_set(validation_set_file_path)
        for directory, subdir, files in os.walk(root_directory):
            subdir.sort()
            files.sort()
            for f in files:
                if f not in validation_set:
                    sorted_events = list()
                    itr_sequence = list()
                    itr_sequence_delayed = list()
                    events = self.get_events_dict_from_file(os.path.join(directory, f))
                    for name, event in sorted(events.iteritems(), key=lambda (n, v): (v.start, n)):
                        sorted_events.append(event)
                        if event.symbol in event_counts:
                            event_counts[event.symbol] += 1
                        else:
                            event_counts[event.symbol] = 1
                    for i in range(0, len(sorted_events) + 1):
                        if i == 0:
                            e = events[sorted_events[i].name]
                            event = ia.AtomicEvent(e.symbol, e.start, e.end)
                            prev = ia.AtomicEvent(None, -sys.maxint, -1)
                        elif i == len(sorted_events):
                            p = events[sorted_events[i - 1].name]
                            event = ia.AtomicEvent(None, sys.maxint - 1, sys.maxint)
                            prev = ia.AtomicEvent(p.symbol, p.start, p.end)
                        else:
                            e = events[sorted_events[i].name]
                            p = events[sorted_events[i - 1].name]
                            event = ia.AtomicEvent(e.symbol, e.start, e.end)
                            prev = ia.AtomicEvent(p.symbol, p.start, p.end)
                            if (e.symbol not in self.transition_events and
                               p.symbol not in self.transition_events):
                                self.nodes[p.symbol].timeout += e.start - p.end
                                timeout_counts[p.symbol] = timeout_counts.get(p.symbol, 0) + 1
                            elif (e.symbol not in self.transition_events and
                                  p.symbol in self.transition_events):
                                self.nodes[p.symbol].timeout += e.start - p.end
                                timeout_counts[p.symbol] = timeout_counts.get(p.symbol, 0) + 1
                        if i < len(sorted_events):
                            if e.symbol not in self.nodes:
                                node = TCGNode(e.symbol, duration=(e.end - e.start))
                                if e.symbol in self.transition_events:
                                    node.is_transition = True
                                self.nodes[e.symbol] = node
                            else:
                                node = self.nodes[e.symbol]
                            node.duration = e.end - e.start + node.duration
                            if i == 0:
                                node.is_start = True
                            elif i == len(sorted_events) - 1:
                                node.is_terminal = True
                        itr = ia.IntervalAlgebra.obtain_itr(event, prev)
                        if (tuple(itr_sequence), (prev.name, event.name)) not in self.transition_cache:
                            self.transition_cache[(tuple(itr_sequence), (prev.name, event.name))] = transition_counter
                            self.itr_cache[transition_counter] = {itr}
                            itr = transition_counter
                            transition_counter += 1
                        else:
                            id = self.transition_cache[(tuple(itr_sequence), (prev.name, event.name))]
                            self.itr_cache[id].add(itr)
                            itr = id
                        itr_sequence.append((prev.name, itr, event.name))
                        # if event.name in self.transition_events:
                        #     event.start += TemporalContextGraph.PERCEPTION_DELAY
                        #     event.end += TemporalContextGraph.PERCEPTION_DELAY
                        #     itr_delayed = ia.IntervalAlgebra.obtain_itr(event, prev)
                        #     itr_sequence_delayed.append((prev.name, itr_delayed, event.name))
                        # else:
                        #     itr_sequence_delayed.append((prev.name, itr, event.name))
                    sequences.append([e.symbol for e in sorted_events])
                    itr_sequences.append(itr_sequence)
                    # itr_sequences.append(itr_sequence_delayed)
        for event, node in self.nodes.iteritems():
            node.duration = float(node.duration) / event_counts[event]
            if event in timeout_counts:
                node.timeout = float(node.timeout) / timeout_counts[event]
            else:
                node.timeout = 0
        return sequences, itr_sequences

    def get_events_dict_from_file(self, file_path):
        """
        Reads the temporal information of a set of events from an input file and stores the
        information in a dict to be returned.
        :param file_path: string indicating the path to an input file with temporal information.
        :return: dict with TCGEvent objects for each of the event in the input file.
        """
        events = dict()
        file_obj = open(file_path, 'r')
        raw_events = file_obj.readlines()
        file_obj.close()
        for e in raw_events:
            event = TCGEvent(e, self)
            if event.start is not None:
                events[event.name] = event
            else:
                events[event.name].end = event.end
        return events

    @staticmethod
    def load_validation_set(validation_set_file_path):
        """
        Loads the names of the files that belong to the VALIDATION set.
        :param validation_set_file_path: string containing the path to the txt file with the
            set of files that belong to the VALIDATION set.
        """
        validation_set = []
        if os.path.exists(validation_set_file_path):
            with open(validation_set_file_path, 'r') as val_set_file:
                validation_set_lines = val_set_file.readlines()
                for line in validation_set_lines:
                    temp_file = line.replace('bags', 'temp_info').replace('.bag', '.txt')
                    validation_set.append(temp_file)
                validation_set.sort()
        return validation_set

    @staticmethod
    def process_itr_sequences(itr_sequences):
        """
        Identifies the order of the n-gram models needed to model the input itr sequences.
        :param itr_sequences: set of training sequences of itr.
        :return: dict containing the all of the events that exist in the training itr sequences and
            the order of the n-gram model needed to perform policy selection for each of them.
        """
        event_grams = dict()
        for sequence in itr_sequences:
            predecessors = dict()
            for i, itr in enumerate(sequence):
                observed_event = itr[2]
                if observed_event is not None:
                    if observed_event not in predecessors:
                        predecessors[observed_event] = list()
                    predecessors[observed_event].append(sequence[:i+1])
            for event, pred_list in predecessors.iteritems():
                if event not in event_grams:
                    event_grams[event] = 1
                min_gram = event_grams[event]
                for mem_size in range(min_gram, len(sequence)):
                    counts = dict()
                    for past in pred_list:
                        evidence = tuple(past[-mem_size:])
                        if evidence not in counts:
                            counts[evidence] = 1
                        else:
                            counts[evidence] += 1
                    if max(counts.values()) == 1:
                        # print('{} -- {}'.format(event, mem_size))
                        event_grams[event] = max(event_grams[event], mem_size)
                        break
        return event_grams

    # OUTPUT METHODS ###############################################################################
    def print_edges(self):
        """
        Prints the set of edges in the Temporal Context Graph
        """
        for _, edge in self.edges.iteritems():
            print(edge)

    def print_nodes(self):
        """
        Prints the set of edges in the Temporal Context Graph
        """
        for _, node in self.nodes.iteritems():
            print(node)

    def output_graph(self, path, open_file=False):
        """
        Generates png and nx representations of the model and saves them to disk
        :param path: string indicating the path to store the network in
        :param open_file: boolean indicating if the file should be opened after being generated.
        """
        self.output_png_file(path)
        if open_file:
            os.system('gnome-open {}.png'.format(path))
        # TODO: save graph to file

    def output_png_file(self, file_path):
        """
        Draws a Temporal Context Graph to the given path
        :param file_path: string indicating the path to save the file in. File extension not needed.
        """
        drawn_edges = set()
        output = "digraph {\n"
        for event, node in self.nodes.iteritems():
            if event not in self.transition_events:
                output += "{} [weight=None];\n".format(event)
        for name, edge in self.edges.iteritems():
            if name not in drawn_edges:
                drawn_edges.add(name)
                output += '{} -> {} [weight=None, label="{}"];\n'.format(
                    edge.start, edge.end, edge.event)
        output += "}"
        dot_file = file_path + '.dot'
        with open(dot_file, "w") as output_file:
            output_file.write(output)
        os.system('dot {} -Tpng -Gsize=10,10\! -o {}.png'.format(dot_file, file_path))

    # POLICY SELECTION #############################################################################
    def initialize_policy_selector(self, initial_state, fail_state, delay=0):
        """
        Initializes the variables needed to perform task execution and inference with a TCG
        :param initial_state: string indicating the initial state of the model.
        :param fail_state: string indicating the default fail state that will be transitioned to
            if the inference process fails.
        :param delay: int indicates the numbers of seconds elapsed before the policy selector is
            initialized.
        """
        self.state = self.nodes[initial_state]
        self.fail_state = self.nodes[fail_state]
        self.timeout = self.state.timeout + self.state.duration + delay
        self.observation = ia.AtomicEvent(None, -1, -1)
        self.sequence = list()
        self.sequence.append(ia.AtomicEvent(initial_state, 0, self.state.duration + delay))

    def process_observation(self, observation, time):
        """
        Process an observation, updates the timeout value and
            identifies a candidate action if needed
        :param observation: string indicating the observation sent from the perception module
        :param time: integer indicating the current time in the execution
        """
        if self.observation.name is None:
            if observation in self.transition_events:
                self.observation.name = observation
                self.observation.start = time
                self.timeout = -1
        elif self.observation.name == observation:
            self.observation.end = time
        else:
            if self.observation.end < 0:
                self.observation.end = time
            if observation in self.transition_events:
                self.observation = ia.AtomicEvent(observation, time, -1)
                self.timeout = -1
            else:
                if self.sequence[-1].name not in self.transition_events:
                    self.sequence.append(self.observation)
                else:
                    self.sequence[-1] = self.observation
                self.timeout = self.nodes[self.observation.name].timeout + time
                self.observation = ia.AtomicEvent(None, -1, -1)

    def evaluate_timeout(self, time):
        if 0 < self.timeout <= time:
            transition = self.sequence[-1].name
            if transition not in self.transition_events:
                transition = TemporalContextGraph.TIMEOUT
            ngram_model = self.ngrams[self.nodes[self.sequence[-1].name].ngram_order]
            itr_sequence = self.generate_itr_sequence(ngram_model.size)
            candidates = set([edge[1] for edge in self.state.edges])
            next_state = ngram_model.evaluate_sequence(itr_sequence, candidates)
            edge = (self.state.event, next_state, transition)
            if edge in self.edges:
                self.timeout = (self.nodes[next_state].timeout +
                                self.nodes[next_state].duration + time)
            else:
                next_state = self.fail_state.event
            self.state = self.nodes[next_state]
            self.sequence.append(ia.AtomicEvent(next_state, time,
                                                time + self.state.duration))
            return next_state
        return None

    def generate_itr_sequence(self, size):
        """
        Generates a itr sequence of the given size using the sequence of events that have
        occurred in the current execution of the task
        :param size: integer indicating the desired length of the sequence
        :return: list containing an itr sequence of the given size
        """
        failure = False
        itr_sequence = list()
        sequence = list(self.sequence)
        for i in range(size - 1):
            sequence.insert(0, None)
        for i, token in enumerate(sequence[:-1]):
            next = sequence[i + 1]
            if token is None and next is None:
                itr = None
            else:
                if token is None:
                    token = ia.AtomicEvent(None, -2, -1)
                itr = ia.IntervalAlgebra.obtain_itr(next, token)
                if (tuple(itr_sequence), (token.name, next.name)) in self.transition_cache:
                    itr_group = self.transition_cache[tuple(itr_sequence), (token.name, next.name)]
                    itr = (token.name, itr_group, next.name)
            itr_sequence.append(itr)
        if failure:
            print(itr_sequence)
        return itr_sequence[-size + 1:]


# TEST METHODS #####################################################################################
def sg_test():
    sg_root = '/home/assistive-robotics/social_greeting_dataset/'
    tcg = TemporalContextGraph(transition_events=['response'])
    tcg.learn_model_from_files(os.path.join(sg_root, 'temp_info/'),
                               validation_file_path=os.path.join(sg_root, 'validation_set.txt')
                               )
    tcg.print_edges()
    tcg.print_nodes()
    print('n-grams:{}'.format(tcg.ngrams.keys()))
    tcg.output_graph('output/sg_graph', False)


def ond_test():
    ond_root = '/home/assistive-robotics/object_naming_dataset/'
    tcg = TemporalContextGraph(transition_events=['incorrect', 'correct', 'visual'])
    tcg.learn_model_from_files(os.path.join(ond_root, 'temp_info/'),
                               validation_file_path=os.path.join(ond_root, 'validation_set.txt')
                               )
    tcg.print_edges()
    tcg.print_nodes()
    print('n-grams:{}'.format(tcg.ngrams.keys()))
    tcg.output_graph('output/ond_graph', False)


def toy_test():
    ond_root = 'input/'
    tcg = TemporalContextGraph(transition_events=['x'])
    tcg.learn_model_from_files(os.path.join(ond_root, ''))
    tcg.print_edges()
    tcg.print_nodes()
    print('n-grams:{}'.format(tcg.ngrams.keys()))
    for order, model in tcg.ngrams.iteritems():
         print(order)
         model.print_model()
    tcg.output_graph('output/toy_graph', False)


if __name__ == '__main__':
    print('# TOY TEST: ######################')
    toy_test()
    # print('\n\n\n# OND TEST: ######################')
    # ond_test()
    # print('\n\n\n# SG TEST: ######################')
    # sg_test()
