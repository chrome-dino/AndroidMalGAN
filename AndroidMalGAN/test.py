import torch
from opcode_ngram_model import BlackBoxDetector
from opcode_ngram_model import NgramGenerator
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVED_MODEL_PATH = '../opcode_ngram_malgan.pth'
BB_SAVED_MODEL_PATH = '../opcode_ngram_blackbox.pth'

blackbox = BlackBoxDetector()
blackbox.load_state_dict(torch.load(BB_SAVED_MODEL_PATH))
blackbox = blackbox.to(DEVICE)
blackbox.eval()

data_malware = np.loadtxt('../malware_ngram.csv', delimiter=',', skiprows=1)
data_malware = (data_malware.astype(np.bool_)).astype(float)

data_benign = np.loadtxt('../benign_ngram.csv', delimiter=',', skiprows=1)
data_benign = (data_benign.astype(np.bool_)).astype(float)

data_benign = data_benign[:, 1:]
data_malware = data_malware[:, 1:]
data_malware = np.array(data_malware)
data_benign = np.array(data_benign)

# convert to tensor
benign = torch.tensor(data_benign).float()
malware = torch.tensor(data_malware).float()

malware = malware.to(DEVICE)
results = blackbox(malware)
results = torch.where(results > 0.5, True, False)
mal = 0
ben = 0
for result in results:
    if result[0]:
        ben += 1
    else:
        mal += 1
print('test 1: Blackbox unmodified malware')
print(f'test set predicted: {str(ben)} benign files and {str(mal)} malicious files\n')

benign = benign.to(DEVICE)
results = blackbox(benign)
results = torch.where(results > 0.5, True, False)
mal = 0
ben = 0
for result in results:
    if result[0]:
        ben += 1
    else:
        mal += 1

print('test 2: Blackbox benign')
print(f'test set predicted: {str(ben)} benign files and {str(mal)} malicious files\n')

generator = NgramGenerator()
generator.load_state_dict(torch.load(SAVED_MODEL_PATH))
generator = generator.to(DEVICE)
generator.eval()

gen_malware = generator(malware)
results = blackbox(gen_malware)
results = torch.where(results > 0.5, True, False)
mal = 0
ben = 0
for result in results:
    if result[0]:
        ben += 1
    else:
        mal += 1

print('test 3: Blackbox malgan malware')
print(f'test set predicted: {str(ben)} benign files and {str(mal)} malicious files\n')