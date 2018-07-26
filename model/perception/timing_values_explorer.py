# Script used to find the mean, max and min duration of the different events that occur in the
# dataset. It also finds the number of correct and incorrect answers shorter than a defined
# threshold. This value is useful to find an appropriate size for the CNN windows
import os

val_set_file = open('/home/assistive-robotics/object_naming_dataset/validation_set.txt', 'r')
validation_set = list()
for line in val_set_file.readlines():
    validation_set.append(line[:-1].replace('bags', 'temp_info').replace('.bag', '.txt'))
val_set_file.close()

fields_count = dict()
fields_duration = dict()
fields_max = dict()
fields_min = dict()
corr_count = 0
inc_count = 0
threshold = 1.5
for directory, subdir, files in os.walk('/home/assistive-robotics/object_naming_dataset/temp_info/'):
    subdir.sort()
    files.sort()
    for f in files:
        if os.path.join(directory, f) not in validation_set:
            o_file = open(os.path.join(directory, f), 'r')
            lines = o_file.readlines()
            o_file.close()
            start = None
            s_name = None
            for line in lines:
                event, time = line.split(' ')
                name = event.split('_')[0]
                if '_s' in event:
                    fields_count[name] = fields_count.get(name, 0) + 1
                    start = float(time)
                    s_name = name
                else:
                    if start is None or name != s_name:
                        print('error')
                    duration = float(time) - start
                    fields_duration[name] = fields_duration.get(name, 0) + duration
                    if duration > fields_max.get(name, 0):
                        fields_max[name] = duration
                    if duration < fields_min.get(name, 100):
                        fields_min[name] = duration
                    if name == 'incorrect' and duration < threshold:
                        inc_count += 1
                    elif name == 'correct' and duration < threshold:
                        corr_count += 1
                    start = None
                    s_name = None

for key, val in fields_count.iteritems():
    print('{}\t{}\t{}\t{}\t{}'.format(key, val, round(fields_duration[key]/val, 2),
                                          fields_min[key], fields_max[key]))
print('c:{} i:{}'.format(corr_count, inc_count))
