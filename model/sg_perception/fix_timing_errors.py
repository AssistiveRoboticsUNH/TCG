import os

def process_real_times(td):
    final_td = dict()
    delete_prompt = False
    mapping = {'noise_0_s': 'command_s', 'noise_0_e': 'command_e',
               'noise_1_s': 'prompt_s', 'noise_1_e': 'prompt_e',
               'audio_0_s': 'response_s', 'audio_0_e': 'response_e',
               'audio_1_s': 'response_s', 'audio_1_e': 'response_e',
               'gesture_0_s': 'response_s', 'gesture_0_e': 'response_e',
               'gesture_1_s': 'response_s', 'gesture_1_e': 'response_e'}
    if td.get('audio_0_s', None) is not None and td.get('audio_1_s', None) is not None:
        del td['audio_1_s']
        del td['audio_1_e']
        td['reward_s'] = td['noise_1_s']
        td['reward_e'] = td['noise_1_e']
        delete_prompt = True
    if td.get('gesture_0_s', None) is not None and td.get('gesture_1_s', None) is not None:
        del td['gesture_1_s']
        del td['gesture_1_e']
        td['reward_s'] = td['noise_1_s']
        td['reward_e'] = td['noise_1_e']
        delete_prompt = True
    if delete_prompt:
        del td['prompt_s']
        del td['prompt_e']
        del td['noise_1_s']
        del td['noise_1_e']
    if td.get('reward_s', None) is not None:
        mapping['abort_s'] = 'reward_s'
        mapping['abort_e'] = 'reward_e'
    for event, time in td.items():
        event = mapping.get(event, event)
        event_name = event.replace('_s', '').replace('_e', '')
        curr_time = final_td.get(event_name, (100000, -1))
        if '_s' in event:
            new_time = (min(time, curr_time[0]), curr_time[1])
        else:
            new_time = (curr_time[0], max(time, curr_time[1]))
        final_td[event_name] = new_time
    # for event in sorted(final_td):
    #     print('{}: {}'.format(event, final_td[event]))
    # print('DEBUG: {}')
    # for event in sorted(td):
    #     print('{}: {}'.format(event, td[event]))
    return final_td

for dir, subdir, files in os.walk('/home/assistive-robotics/social_greeting_dataset/old_temp_info'):
    subdir.sort()
    files.sort()
    for f in files:
        td = dict()
        output = ''
        with open(os.path.join(dir, f), 'r') as infile:
            lines = infile.readlines()
        for line in lines:
            info = line.split(' ')
            td[info[0]] = float(info[1][:-1])
        td = process_real_times(td)
        for event, times in sorted(td.iteritems(), key=lambda v: v[1]):
            output += '{}_s {}\n'.format(event, times[0])
            output += '{}_e {}\n'.format(event, times[1])
        with open(os.path.join(dir.replace('old_', ''), f), 'w') as outfile:
            outfile.write(output)
