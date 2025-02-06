import torch
from intents_inject import inject as intent_inject
from permissions_inject import inject as permission_inject
from ngram_inject import inject as ngrams_inject
from apis_inject import inject as api_inject
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def daisy_chain_attack(file_path='', n_count=5, blackbox=''):
    f_list = list(os.path.split(file_path))
    f_name = f_list[-1]
    f_list[-1] = 'modified_' + f_name
    file_path_mod = os.path.join(*f_list)
    # inject ngram
    ngrams_inject(file_path, copy_file=True, n_count=n_count, blackbox=blackbox)
    # inject permissions
    permission_inject(file_path_mod, copy_file=False, blackbox=blackbox)
    # inject intent
    intent_inject(file_path_mod, copy_file=False, blackbox=blackbox)
    # inject api
    api_inject(file_path_mod, copy_file=False, blackbox=blackbox)
    return file_path_mod
