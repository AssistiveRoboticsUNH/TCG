import os

lines = ''

for dir, subdir, files in os.walk('/home/assistive-robotics/social_greeting_dataset/old_temp_info'):
    subdir.sort()
    files.sort()
    for f in files:
        with open(os.path.join(dir, f), 'r') as infile:
            lines = infile.readlines()
        if not os.path.exists(os.path.join(dir.replace('old_', ''))):
            os.mkdir(os.path.join(dir.replace('old_', '')))
        with open(os.path.join(dir.replace('old_', ''), f), 'w') as outfile:
            for line in lines:
                if 'noise' not in line:
                    outfile.write(line)
