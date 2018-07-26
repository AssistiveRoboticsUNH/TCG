# Script used to rename bag files

from configuration import *

gc = 0
for directory, subdir, files in os.walk(ond_rosbags_path):
    counter = 0
    files.sort()
    for f in files:
        if f.endswith('.bag'):
            gc += 1
            if 'failA' in directory:
                prefix = 'fa'
            elif 'failB' in directory:
                prefix = 'fb'
            elif 'successA' in directory:
                prefix = 'sa'
            elif 'successB' in directory:
                prefix = 'sb'
            elif 'successC' in directory:
                prefix = 'sc'
            new_name = '{}/{}_{}.bag'.format(directory, prefix, counter)
            old_name = '{}/{}'.format(directory, f)
            # os.rename(old_name, new_name)
            print('{}, {}'.format(new_name, old_name))
            counter += 1
print(gc)
