# 8.2.2 Downloading the data

#
# Listing 8.6 Copying images to training, validation, and test directories
import os, shutiil, pathlib

# Path to original directory where original dataset was uncompressed
original_dir = "~/src/cats_vs_dogs/"
# Directory where we will store our smaller dataset
new_base_dir = pathlib("cats_vs_dogs_small")

# Utility function to copy cat (and dog) images from index start_index to index end_index 
# to the subdirectory new_base_dir/{subset_name}/cat (and /dog).
# The "subset_name" will be either "train", "validation", or "test".
def make_subset(subset_name, start_index, end_index):
    for category in ("cat", "dog"):
