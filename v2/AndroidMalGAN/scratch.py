import os
import re
root_dir = '../samples/benign_samples/decompiled'
count = 0
current_hash = ''
t = []
for root, dirs, files in os.walk(root_dir):  # Scanning through each file in each subdirectory
    for file in files:
        if file.endswith(".smali"):
            file_dest = os.path.join(root, file)
            md5_hash = re.findall(r"([a-fA-F\d]{32})", file_dest)[0]

            if md5_hash != current_hash:
                count += 1
                current_hash = md5_hash
                # print(file_dest + ' ' + str(count))
                t.append(md5_hash)

print(len(t))
samples = []
# final = []
sample_md5s = []

dirs = [item[0] for item in os.walk(root_dir)]
# print(dirs)
# exit(-1)
for sub_dir in dirs:
    # print(sub_dir)
    if md5_hash := re.findall(r"([a-fA-F\d]{32})", sub_dir):
        if md5_hash[0] not in sample_md5s:
            if not os.listdir(sub_dir):
                continue
            sample_md5s.append(md5_hash[0])
            samples.append(sub_dir)
print(len(samples))
