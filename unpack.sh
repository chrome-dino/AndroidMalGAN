#!/bin/bash

cd samples
for file in ./samples/*.apk; do md5=($(md5sum file));apktool d -f $file -o $md5; done