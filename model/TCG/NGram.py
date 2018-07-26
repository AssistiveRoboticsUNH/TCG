class Gram:
    """Class that represents a set of contiguous tokens"""
    def __init__(self, tuple_obj):
        """
        Constructor
        :param tuple_obj: set of tokens that forms the gram
        """
        self.tuple = tuple_obj
        self.count = 0.
        self.successors = dict()

    def __repr__(self):
        """String representation of Gram object"""
        return 'c:{} - t:{}'.format(self.count, self.tuple)


class NGramModel:
    """Class created to abstract and learn an n-gram model"""

    def __init__(self, size):
        """
        Constructor
        :param size: value of the n parameter of the model
        """
        self.size = size
        self.grams = dict()

    def learn_model(self, sequences):
        """
        Learns and stores an n-gram model given a set of sequences
        :param sequences: set of sequences that will be used to learn the n-gram model.
                          Each sequence consists of a list of tokens
        """
        for seq in sequences:
            for i in range(0, self.size - 1):
                seq.insert(0, None)
                seq.append(None)
            for i in range(0, len(seq) - (self.size - 1)):
                tokens = tuple(seq[i: i + self.size - 1])
                next_token = seq[i + self.size - 1]
                gram = self.grams.get(tokens, None)
                if gram is None:
                    gram = Gram(tokens)
                    self.grams[tokens] = gram
                if next_token is not None:
                    gram.successors[next_token[-1]] = gram.successors.get(next_token[-1], 0) + 1
                else:
                    gram.successors[next_token] = gram.successors.get(next_token, 0) + 1
                gram.count += 1

    def print_model(self):
        """Prints grams and probabilities"""
        for tokens, gram in self.grams.iteritems():
            print('{}'.format(tokens))
            for successor, count in gram.successors.iteritems():
                probability = count / gram.count
                print('\t{} {:.3f}'.format(successor, probability))

    def evaluate_sequence(self, sequence, candidates):
        """
        Returns the grams with the highest likelihood given the input sequence
        :param sequence: list containing a sequence to be evaluated
        :param candidates: list of strings indicating the possible future states
        :return: candidate gram with the highest likelihood given the input sequence
        """
        gram = self.grams.get(tuple(sequence), None)
        max_prob = 0
        next_state = None
        if gram is not None:
            for successor, count in gram.successors.iteritems():
                prob = count / gram.count
                if prob > max_prob and successor in candidates:
                    max_prob = count / gram.count
                    next_state = successor
        return next_state


# TEST ROUTINES
def toy_test():
    seqs = [['COM', 'PMT', 'PMT', 'PMT', 'PMT', 'PMT', 'ABT'],
            ['COM', 'COR', 'REW'],
            ['COM', 'PMT', 'INC', 'PMT', 'INC', 'PMT', 'COR', 'REW'],
            ['COM', 'INC', 'PMT', 'VIS', 'ABT']]
    n = NGramModel(3)
    n.learn_model(seqs)
    n.print_model()


if __name__ == '__main__':
    print('Running toy test:')
    toy_test()
