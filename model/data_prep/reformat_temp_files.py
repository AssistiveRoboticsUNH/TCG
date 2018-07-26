import os

root_path = "C:\users\eccar\Cortana\Grad School\Box Sync\Thesis\ond_audio\\temp_info"
file_count = 0

for dir, subdir, files in os.walk(root_path):
    for f in files:
        file_name = f
        file_count += 1
        # READ FILES
        f = open(os.path.join(dir, f), "r")
        temp_info = f.readlines()
        out_info = dict()
        prompt_count = 0
        for line in temp_info:
            line_info = line.split(" ")
            line_info[1] = "{0:.2f}".format(float(line_info[1]))
            if "command" in line_info[0] and out_info.get("command_s", None) is None:
                out_info["command_s"] = line_info[0] + "_s " + line_info[1]
                out_info["command_e"] = line_info[0] + "_e "
            elif "prompt" in line_info[0]:
                prompt_name = "prompt_{}".format(prompt_count)
                out_info[prompt_name + "_s"] = prompt_name + "_s " + line_info[1]
                out_info[prompt_name + "_e"] = prompt_name + "_e"
                prompt_count += 1
            elif "abort" in line_info[0]:
                out_info["abort_s"] = line_info[0] + "_s " + line_info[1]
                out_info["abort_e"] = line_info[0] + "_e "
            elif "reward" in line_info[0]:
                out_info["reward_s"] = line_info[0] + "_s " + line_info[1]
                out_info["reward_e"] = line_info[0] + "_e "
            elif "command" in line_info[0] or 'stop' in line_info[0] or 'start' in line_info[0]:
                pass
            else:
                print(line_info[0])
        f.close()

        # WRITE FILES
        f = open(os.path.join(dir, file_name), "w")
        f.write("{}\n".format(out_info["command_s"]))
        f.write("{}\n".format(out_info["command_e"]))
        for i in range(0, prompt_count):
            prompt_name = "prompt_{}".format(i)
            f.write("{}\n".format(out_info[prompt_name + "_s"]))
            f.write("{}\n".format(out_info[prompt_name + "_e"]))
        if out_info.get("abort_s", None) is not None:
            f.write("{}\n".format(out_info["abort_s"]))
            f.write("{}\n".format(out_info["abort_e"]))
        elif out_info.get("reward_s", None) is not None:
            f.write("{}\n".format(out_info["reward_s"]))
            f.write("{}\n".format(out_info["reward_e"]))
        for i in range(0, prompt_count):
            f.write("incorrect_{}_s \n".format(i))
            f.write("incorrect_{}_e \n".format(i))
        if out_info.get("abort_s", None) is not None:
            f.write("incorrect_{}_s \n".format(prompt_count))
            f.write("incorrect_{}_s \n".format(prompt_count))
        elif out_info.get("reward_s", None) is not None:
            f.write("correct_s \n")
            f.write("correct_e \n")
        # print('')
        f.close()
print("{}".format(file_count))
