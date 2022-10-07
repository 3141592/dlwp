#!/bin/bash
# input_dir = "/root/src/data/images/"
# target_dir = "/root/src/data/annotations/trimaps/"

# WGET the data
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz -P /tmp/
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz -P /tmp/

# Extract to data directory
tar xvf /tmp/images.tar.gz -C /root/src/data/
tar xvf /tmp/annotations.tar.gz -C /root/src/data/

# Delete the archives
rm -rf /tmp/images.tar.gz*
rm -rf /tmp/annotations.tar.gz*

