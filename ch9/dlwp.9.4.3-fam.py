# 9.4.3 Visualizing heatmaps of class activation
print("9.4.3 Visualizing heatmaps of class activation")
# Suppress warnings
import os, pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 
# Listing 9.20 Loading the Xception network with pretrained weights
print("Listing 9.20 Loading the Xception network with pretrained weights")
import tensorflow.keras as keras
import numpy as np

model = keras.applications.xception.Xception(weights="imagenet")

#
# Listing 9.21 Preprocessing an input image for Xecption
print("Listing 9.21 Preprocessing an input image for Xecption")

def process_image(no):
    if(no==0):
        # [('n03026506', 'Christmas_stocking', 0.3573506), ('n03188531', 'diaper', 0.05843441), ('n03877472', 'pajama', 0.048327956)]
        img_path = "/mnt/d/IMG_0486.jpg"
    elif(no==1):
        # [('n09421951', 'sandbar', 0.85636437), ('n04371430', 'swimming_trunks', 0.025900142), ('n03623198', 'knee_pad', 0.0068305633)]
        img_path = "/mnt/d/IMG020.jpg"
    elif(no==2):
        # [('n03201208', 'dining_table', 0.20655847), ('n03179701', 'desk', 0.18680723), ('n04522168', 'vase', 0.18470599)]
        img_path = "/mnt/d/DSCN2287.jpg"
    elif(no==3):
        # [('n02123597', 'Siamese_cat', 0.64078176), ('n02127052', 'lynx', 0.14561251), ('n02124075', 'Egyptian_cat', 0.05403875)]
        img_path = keras.utils.get_file(fname="cat.jpg", origin="https://img-datasets.s3.amazonaws.com/cat.jpg")
    elif(no==4):
        # [('n02108422', 'bull_mastiff', 0.47068202), ('n02106550', 'Rottweiler', 0.092656404), ('n02088466', 'bloodhound', 0.034463808)]
        img_path = "/root/src/data/cats_vs_dogs_small/test/dog/dog.2499.jpg"
    elif(no==5):
        # [('n02109047', 'Great_Dane', 0.9922776), ('n02100236', 'German_short-haired_pointer', 0.001078376), ('n02092339', 'Weimaraner', 0.0007794483)]
        img_path = "/mnt/d/great.dane.jpg"
    elif(no==6):
        # [('n02109047', 'Great_Dane', 0.9916483), ('n02100236', 'German_short-haired_pointer', 0.00055604626), ('n02092339', 'Weimaraner', 0.00041812126)]
        img_path = "/mnt/d/great.dane.puppy.jpg"
    elif(no==7):
        # [('n06596364', 'comic_book', 0.8340095), ('n07248320', 'book_jacket', 0.011623779), ('n03598930', 'jigsaw_puzzle', 0.0057839802)]
        img_path = "/mnt/d/Scoobydoo.jpg"
    elif(no==8):
        # [('n06596364', 'comic_book', 0.53293544), ('n03598930', 'jigsaw_puzzle', 0.036701314), ('n04116512', 'rubber_eraser', 0.021377431)]
        img_path = "/mnt/d/scooby.jpg"
    elif(no==9):
        # [('n02085620', 'Chihuahua', 0.15602025), ('n02091032', 'Italian_greyhound', 0.15489765), ('n02087046', 'toy_terrier', 0.15129827)]
        img_path = "/mnt/d/photodog1.jpg"
    elif(no==10):
        # [('n02087394', 'Rhodesian_ridgeback', 0.6768997), ('n02109047', 'Great_Dane', 0.07607841), ('n02090379', 'redbone', 0.06192694)]
        img_path = "/mnt/d/dog.drawing.1.jpg"
    elif(no==11):
        # [('n02088094', 'Afghan_hound', 0.3908979), ('n02091831', 'Saluki', 0.15265788), ('n02100735', 'English_setter', 0.016079867)]
        img_path = "/mnt/d/dog.drawing.2.jpg"
    elif(no==12):
        # [('n01698640', 'American_alligator', 0.89597297), ('n01697457', 'African_crocodile', 0.062412985), ('n01665541', 'leatherback_turtle', 0.0003912723)]
        img_path = "/mnt/d/aligator.1.png"
    elif(no==13):
        # [('n04350905', 'suit', 0.63127065), ('n04591157', 'Windsor_tie', 0.2877923), ('n02865351', 'bolo_tie', 0.015513565)]
        img_path = "/mnt/d/tlj.jpg"
    elif(no==14):
        # [('n03710637', 'maillot', 0.56471676), ('n03710721', 'maillot', 0.083704114), ('n03255030', 'dumbbell', 0.05399074)]
        img_path = "/mnt/d/superman.jpg"
    elif(no==15):
        # [('n03255030', 'dumbbell', 0.84658635), ('n02790996', 'barbell', 0.13791782), ('n04372370', 'switch', 0.0014434928)]
        img_path = "/mnt/d/dumbbell.jpg"

    return img_path

image_number = 7
img_path = process_image(image_number)

def get_img_array(img_path, target_size):
    # Return a Python Imaging Library (PIL) image of size 299 x 299.
    img = keras.utils.load_img(img_path, target_size=target_size)
    # Return a float32 NumPy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(img)
    # Add a dimension to transform the array into a batch of size (1, 299, 299, 3).
    array = np.expand_dims(array, axis=0)
    # Preprocess the batch (this does channel-wise color normalization).
    array = keras.applications.xception.preprocess_input(array)
    return array

img_array = get_img_array(img_path, target_size=(299, 299))

# Run the pretrained network on the image and deocde its prediction vector
print("Run the pretrained network on the image and deocde its prediction vector")
preds = model.predict(img_array)
print(keras.applications.xception.decode_predictions(preds, top=3)[0])

#
# Listing 9.22 Setting up a model that returns the last convolutional output
print("Listing 9.22 Setting up a model that returns the last convolutional output")
last_conv_layer_name = "block14_sepconv2_act"
classifier_layer_names = [
        "avg_pool",
        "predictions"
        ]
last_conv_layer = model.get_layer(last_conv_layer_name)
last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

#
# Listing 9.23 Reapplying the classifier on top of the last convolutional output
print("Listing 9.23 Reapplying the classifier on top of the last convolutional output")
classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
x = classifier_input
for layer_name in classifier_layer_names:
    x = model.get_layer(layer_name)(x)
classifier_model = keras.Model(classifier_input, x)

# Listing 9.24 Retrieving the gradients of the top predicted class
print("Listing 9.24 Retrieving the gradients of the top predicted class")
import tensorflow as tf

with tf.GradientTape() as tape:
    # Compute activation of the last conv layer and make the tape watch it.
    last_conv_layer_output = last_conv_layer_model(img_array)
    tape.watch(last_conv_layer_output)
    # Retrieve the activation channel corresponding to the top predicted class
    preds = classifier_model(last_conv_layer_output)
    top_pred_index = tf.argmax(preds[0])
    top_class_channel = preds[:, top_pred_index]

# This is the gradient of the top predicted class with regard
# to the output feature map of the last convolutional layer.
grads = tape.gradient(top_class_channel, last_conv_layer_output)

#
# Listing 9.25 Gradient pooling and channel-importance weighing
print("Listing 9.25 Gradient pooling and channel-importance weighing")

# This is a vector where each entry is the mean intensity of the gradient for a given channel.
# It quantifies the importance of each channel with regard to the top predicted class.
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
last_conv_layer_output = last_conv_layer_output.numpy()[0]

# Multiple each channel in the output of the last convolutional layer
# by "how important this channel is".
for i in range(pooled_grads.shape[-1]):
    last_conv_layer_output[:, :, i] *= pooled_grads[i]

# The channel-wise mean of the resulting feature map is our heatmap of class activation.
heatmap = np.mean(last_conv_layer_output, axis=-1)

#
# Listing 9.26 Heatmap post-processing
print("Listing 9.26 Heatmap post-processing")
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

#
# Listing 9.27 Superimposing the heatmap on the original picture
print("Listing 9.27 Superimposing the heatmap on the original picture")
import matplotlib.cm as cm

img = keras.utils.load_img(img_path)
img = keras.utils.img_to_array(img)

# Rescale the heatmap to 0-255.
print("Rescale the heatmap to 0-255.")
heatmap = np.uint8(255 * heatmap)

# Use the "jet" colormap to recolorize the heatmap.
print("Use the \"jet\" colormap to recolorize the heatmap.")
jet = cm.get_cmap("jet")
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[heatmap]

# Create an image that contains the recolorized heatmap
print("Create an image that contains the recolorized heatmap")
jet_heatmap = keras.utils.array_to_img(jet_heatmap)
jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
jet_heatmap = keras.utils.img_to_array(jet_heatmap)

# Superimpose the heatmap and the original image, with the heatmap at 40% opacity.
print("Superimpose the heatmap and the original image, with the heatmap at 40% opacity.")
superimposed_img = jet_heatmap * 0.6 + img
superimposed_img = keras.utils.array_to_img(superimposed_img)

import io
basename_without_ext = os.path.splitext(os.path.basename(img_path))[0]
print("Saving image.")
save_path = "/mnt/d/" + basename_without_ext + "_cam.jpg"
superimposed_img.save(save_path)
