import numpy as np
import torch
data_malware = np.loadtxt('malware.csv', delimiter=',', skiprows=1)
# with open('benign.csv') as f:                                                                   
#     ncols = len(f.readline().split(','))                                                        
data_benign = np.loadtxt('benign.csv', delimiter=',', skiprows=1)

labels_benign = data_benign[:, 0]
data_benign = data_benign[:, 1:]

labels_malware = data_malware[:, 0]
data_malware = data_malware[:, 1:]

# normalize the data to a range of [-1 1] (b/c tanh output)                                       
data_benign = data_benign / np.max(data_benign)
data_benign = 2 * data_benign - 1

data_malware2 = data_malware / np.max(data_malware)
data_malware2 = 2 * data_malware2 - 1

# norm = torch.transforms.normalize()
mean, std = np.mean(data_malware), np.std(data_malware)
data_malware_norm = [(element - mean)/std for element in data_malware]

data_tensor_benign = torch.tensor(data_benign).float()
data_malware_norm = np.array(data_malware_norm)
data_malware = np.array(data_malware)
# print(data_malware[0])
# print(data_malware_norm[0])
data_tensor_malware = torch.tensor(data_malware_norm).float()
data_tensor_malware2 = torch.tensor(data_malware).float()
data_tensor_malware3 = torch.tensor(data_malware2).float()
# print(data_tensor_malware[0])
# print(data_tensor_malware2[0])
# print(data_tensor_malware3[0])