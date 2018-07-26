import numpy as np


class IntervalTemporalRelation:
    """Class created to store the attributes of the different interval temporal relations"""

    def __init__(self, name, abbreviation, code):
        """
        Constructor
        :param name: interval temporal relationship name
        :param abbreviation: string code for the itr
        :param code: integer code for the itr
        """
        self.name = name
        self.abbreviation = abbreviation
        self.code = code

    def __repr__(self):
        """
        Representation
        :return: Abbreviation of the itr
        """
        return self.abbreviation


class AtomicEvent:
    """Class created to store the attributes of atomic events"""

    def __init__(self, name, start_time, end_time):
        """
        Constructor
        :param name: interval temporal relationship name
        :param start_time: event start time
        :param end_time: event end time
        """
        self.name = name
        self.start = start_time
        self.end = end_time

    def __repr__(self):
        """String representation of an AtomicEvent"""
        return '{}, {:.2f}-{:.2f}'.format(self.name, self.start, self.end)


class IntervalAlgebra:
    """
    Class created to abstract and learn the interval temporal relations between a set of events
    """

    """Map from temporal distances to interval temporal relationship"""
    TEMP_DISTANCE_ITR = {(-1., -1., -1., -1.): IntervalTemporalRelation('before', 'b', 0),
                         (1., 1., 1., 1.): IntervalTemporalRelation('before_i', 'bi', 1),
                         (1., -1., -1., 1.): IntervalTemporalRelation('during', 'd', 2),
                         (-1., 1., -1., 1.): IntervalTemporalRelation('during_i', 'di', 3),
                         (-1., -1., -1., 1.): IntervalTemporalRelation('overlaps', 'o', 4),
                         (1., 1., -1., 1.): IntervalTemporalRelation('overlaps_i', 'oi', 5),
                         (-1., -1., -1., 0.): IntervalTemporalRelation('meets', 'm', 6),
                         (1., 1., 0., 1.): IntervalTemporalRelation('meets_i', 'mi', 7),
                         (0., -1., -1., 1.): IntervalTemporalRelation('starts', 's', 8),
                         (0., 1., -1., 1.): IntervalTemporalRelation('starts_i', 'si', 9),
                         (1., 0., -1., 1.): IntervalTemporalRelation('finishes', 'f', 10),
                         (-1., 0., -1., 1.): IntervalTemporalRelation('finishes_i', 'fi', 11),
                         (0., 0., -1., 1.): IntervalTemporalRelation('equal', 'e', 12)}

    def __init__(self):
        """Constructor"""
        pass

    def learn_itr_set(self, events):
        """
        Learns the interval temporal relationships between a set of events
        :param events: list of atomic events
        """
        itr_map = dict()
        for event_a in events:
            for event_b in events:
                itr_set = set()
                if not event_a == event_b:
                    relation = self.obtain_itr(event_b, event_a)
                    itr_set.add(relation)
                event_tuple = (event_a.name, event_b.name)
                itr_map[event_tuple] = sorted(itr_set.union(itr_map.get(event_tuple, [])))
        return itr_map

    @staticmethod
    def obtain_itr(event_a, event_b):
        """
        Obtains the interval temporal relationships between two events
        :param event_a: atomic event A
        :param event_b: atomic event B
        """
        # calculate temporal distance
        temp_distance = (np.sign(event_b.start - event_a.start), np.sign(event_b.end - event_a.end),
                         np.sign(event_b.start - event_a.end), np.sign(event_b.end - event_a.start))
        return IntervalAlgebra.TEMP_DISTANCE_ITR[temp_distance]


# TESTING PROCEDURES
def test_obtain_itr():
    """Simple test that obtains the ITR between two events"""
    ia = IntervalAlgebra()
    event_a = AtomicEvent('a', 100, 120)
    event_b = AtomicEvent('b', 0, 10)
    itr = ia.obtain_itr(event_b, event_a)
    print('{}{{{}}}{}'.format(event_a.name, itr.abbreviation, event_b.name))


def test_learn_itr_set():
    """Simple test that obtains the set of ITRs that exist between a set of events"""
    ia = IntervalAlgebra()
    event_a = AtomicEvent('a', 100, 120)
    event_a2 = AtomicEvent('a', 0, 0.5)
    event_b = AtomicEvent('b', 0, 10)
    event_c = AtomicEvent('c', 1, 121)
    events = [event_a, event_a2, event_b, event_c]
    itr = ia.learn_itr_set(events)
    for key in sorted(itr.keys()):
        itr_list = ''
        for rel in itr[key]:
            itr_list += ',' + rel.abbreviation
        print('{} {}{{{}}}{}'.format(len(itr[key]), key[0], itr_list[1:], key[1]))


if __name__ == '__main__':
    print('Running toy tests')
    print('Testing obtain itr')
    test_obtain_itr()
    print('\nTesting learn itr set')
    test_learn_itr_set()
