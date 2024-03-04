import os
import subprocess
import re
import ntlk
from ntlk import word_tokenize
from ntlk.util import ngrams

SAMPLE_DIR = ''

def extract_permissions():
    print('')

def extract_ngrams(filename, n=2):
    # java -Xmx512m -jar path_to_baksmali.jar dexfile.dex
    with open(filename) as f:
        token = ntlk.word_tokenize(f.read())
        ngram_list = ngrams(token, n)

def extract_opcodes():
    print('')

def extract_strings(filename, min=4):
    with open(filename) as f:
        return re.findall("[^\x00-\x1F\x7F-\xFF]{%s,}" % str(min), f.read())
    return False

def extract_imports():
    print('')

def extract_cfg():
    print('')

def extract_api():
    print('')


subprocess.run(['sh', './unpack.sh'])
