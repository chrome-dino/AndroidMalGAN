import argparse
import ngram_inject
import opcode_ngram_model


def train():
    opcode_ngram_model.train()
    return


def inject_attack(input_file):
    ngram_inject.inject(input_file)
    return


def main():
    # cmd line args
    parser = argparse.ArgumentParser(description='Android MalGAN tool')
    parser.add_argument("-m", "--mode", help="Mode to run tool in. Choose from: train, inject", required=True)
    parser.add_argument("-i", "--input_file", help="File to inject into", required=False)
    args = parser.parse_args()

    mode_whitelist = ['train', 'inject']

    if args.mode not in mode_whitelist:
        print('Tool mode must be one of: train, inject')
        exit(-1)
    if args.mode == 'inject' and not args.input_file:
        print('Inject mode requires an input_file arg')
        exit(-1)

    if args.mode == 'train':
        train()
    if args.mode == 'inject':
        inject_attack(args.input_file)
