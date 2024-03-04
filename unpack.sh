#!/bin/bash

cd samples
for file in ./samples/*.apk; do apktool d -f $file; done