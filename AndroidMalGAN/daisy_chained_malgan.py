import torch
from permissions_model import PermissionsGenerator
from apis_model import ApisGenerator
from intents_model import IntentsGenerator
from opcode_ngram_model import NgramGenerator

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def daisy_chain_attack(data_malware):
    intents_generator = IntentsGenerator()
    intents_generator.load_state_dict(torch.load('./intents_malgan.pth')).to(DEVICE)
    intents_generator.eval()
    api_generator = ApisGenerator()
    api_generator.load_state_dict(torch.load('./opcode_ngram_malgan.pth')).to(DEVICE)
    api_generator.eval()
    permissions_generator = PermissionsGenerator()
    permissions_generator.load_state_dict(torch.load('./permissions_malgan.pth')).to(DEVICE)
    permissions_generator.eval()
    ngram_generator = NgramGenerator()
    ngram_generator.load_state_dict(torch.load('./opcode_ngram_malgan.pth')).to(DEVICE)
    ngram_generator.eval()
    test_data_malware = data_malware.to(DEVICE)

    gen_malware = permissions_generator(test_data_malware)
    binarized_gen_malware = torch.where(gen_malware > 0.5, 1.0, 0.0)
    binarized_gen_malware_logical_or = torch.logical_or(test_data_malware, binarized_gen_malware).float()
    gen_malware = binarized_gen_malware_logical_or.to(DEVICE)

    gen_malware = api_generator(gen_malware)
    binarized_gen_malware = torch.where(gen_malware > 0.5, 1.0, 0.0)
    binarized_gen_malware_logical_or = torch.logical_or(test_data_malware, binarized_gen_malware).float()
    gen_malware = binarized_gen_malware_logical_or.to(DEVICE)

    gen_malware = intents_generator(gen_malware)
    binarized_gen_malware = torch.where(gen_malware > 0.5, 1.0, 0.0)
    binarized_gen_malware_logical_or = torch.logical_or(test_data_malware, binarized_gen_malware).float()
    gen_malware = binarized_gen_malware_logical_or.to(DEVICE)

    gen_malware = ngram_generator(gen_malware)
    binarized_gen_malware = torch.where(gen_malware > 0.5, 1.0, 0.0)
    binarized_gen_malware_logical_or = torch.logical_or(test_data_malware, binarized_gen_malware).float()
    gen_malware = binarized_gen_malware_logical_or.to(DEVICE)

    return gen_malware
