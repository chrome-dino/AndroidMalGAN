#!/bin/bash

cd samples
cd benign_samples
files=$(find . -type f -name "*.apk");
decompiled="decompiled/"
for file in $files; do md5=$decompiled$(md5sum $file); $(echo $md5); $(apktool d -f $file -o $md5); done;

cd ../malware_samples
files=$(find . -type f -name "*.apk");
for file in $files; do md5=$decompiled$(md5sum $file); $(echo $md5); $(apktool d -f $file -o $md5); done;
