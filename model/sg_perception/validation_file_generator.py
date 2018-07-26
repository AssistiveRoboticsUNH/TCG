import os

output = ''

for dir, subdir, files in os.walk(
        '/home/assistive-robotics/social_greeting_dataset/ITBN_tfrecords'):
    subdir.sort()
    files.sort()
    for f in files:
        if 'validation' in f:
            f = os.path.join(dir.replace('ITBN_tfrecords', 'bags'),
                             f.replace('.tfrecord', '.bag'))
            output += f + '\n'

print(output)
with open('/home/assistive-robotics/social_greeting_dataset/validation_set.txt', 'w') as outfile:
    outfile.write(output)
