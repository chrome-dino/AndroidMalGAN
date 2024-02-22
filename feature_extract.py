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

for filename in os.listdir(SAMPLE_DIR):
    f = os.path.join(SAMPLE_DIR, filename)
    unzip_dir = f + '_unzip'

    subprocess.run(["apk", "d", f, "-o", unzip_dir]) 
    # if not os.path.exists(unzip_dir):
    #     os.makedirs(unzip_dir)
    #     if os.path.isfile(f):
    #         with zipfile.ZipFile(f, 'f') as zip_f:
    #             zip_f.extractall(unzip_dir)
    #         unzip_dir
