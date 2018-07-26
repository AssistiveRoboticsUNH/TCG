"""
Generates a txt file containing the path of the files of the VALIDATION set.
"""
import random
from configuration import *

validation_set_share = 0.25

types = dict()
for dir, subdir, files in os.walk(os.path.join(ond_rosbags_path, 'bags')):
    subdir.sort()
    files.sort()
    for f in files:
        code = f.split('_')[0]
        file_set = types.get(code, None)
        if file_set is None:
            file_set = list()
            types[code] = file_set
        file_set.append(os.path.join(dir, f))

validation_set = list()
training_set = list()
for code, file_set in types.iteritems():
    count = len(file_set)
    validation_count = int(count * validation_set_share)
    validation_set += random.sample(file_set, validation_count)
    training_set += [f for f in file_set if f not in validation_set]
    print('{} t:{} tr:{} va:{}'.format(code, count, len(training_set), len(validation_set)))

out_file = open(os.path.join(ond_rosbags_path, 'validation_set.txt'), 'w')
validation_set.sort()
for f in validation_set:
    out_file.write(f + '\n')
out_file.close()
