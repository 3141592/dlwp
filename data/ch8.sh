#!/bin/bash
kaggle competitions download -c dogs-vs-cats
mv dogs-vs-cats.zip /tmp/

mkdir /root/src/data/dogs-vs-cats/
unzip /tmp/dogs-vs-cats.zip -d /root/src/data/dogs-vs-cats/
unzip /root/src/data/dogs-vs-cats/train.zip -d /root/src/data/dogs-vs-cats/

rm /tmp/dogs-vs-cats.zip
