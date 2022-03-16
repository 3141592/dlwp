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
#img_path = keras.utils.get_file(
#        fname="elephant.jpg",
#        origin="https://img-datasets.s3.amazonaws.com/elephant.jpg")
# [('n03026506', 'Christmas_stocking', 0.3573506), ('n03188531', 'diaper', 0.05843441), ('n03877472', 'pajama', 0.048327956)]
img_path = "/mnt/d/IMG_0486.jpg"
# [('n09421951', 'sandbar', 0.85636437), ('n04371430', 'swimming_trunks', 0.025900142), ('n03623198', 'knee_pad', 0.0068305633)]
img_path = "/mnt/d/IMG020.jpg"
# [('n03201208', 'dining_table', 0.20655847), ('n03179701', 'desk', 0.18680723), ('n04522168', 'vase', 0.18470599)]
img_path = "/mnt/d/DSCN2287.jpg"
# [('n02123597', 'Siamese_cat', 0.64078176), ('n02127052', 'lynx', 0.14561251), ('n02124075', 'Egyptian_cat', 0.05403875)]
img_path = keras.utils.get_file(fname="cat.jpg", origin="https://img-datasets.s3.amazonaws.com/cat.jpg")
# [('n02108422', 'bull_mastiff', 0.47068202), ('n02106550', 'Rottweiler', 0.092656404), ('n02088466', 'bloodhound', 0.034463808)]
img_path = "/root/src/cats_vs_dogs_small/test/dog/dog.2499.jpg"
# [('n02109047', 'Great_Dane', 0.9922776), ('n02100236', 'German_short-haired_pointer', 0.001078376), ('n02092339', 'Weimaraner', 0.0007794483)]
img_path = "/mnt/d/great.dane.jpg"
# [('n02109047', 'Great_Dane', 0.9916483), ('n02100236', 'German_short-haired_pointer', 0.00055604626), ('n02092339', 'Weimaraner', 0.00041812126)]
img_path = "/mnt/d/great.dane.puppy.jpg"
# [('n06596364', 'comic_book', 0.8340095), ('n07248320', 'book_jacket', 0.011623779), ('n03598930', 'jigsaw_puzzle', 0.0057839802)]
img_path = "/mnt/d/Scoobydoo.jpg"
# [('n06596364', 'comic_book', 0.53293544), ('n03598930', 'jigsaw_puzzle', 0.036701314), ('n04116512', 'rubber_eraser', 0.021377431)]
img_path = "/mnt/d/scooby.jpg"
# [('n02085620', 'Chihuahua', 0.15602025), ('n02091032', 'Italian_greyhound', 0.15489765), ('n02087046', 'toy_terrier', 0.15129827)]
img_path = "/mnt/d/photodog1.jpg"
# [('n02087394', 'Rhodesian_ridgeback', 0.6768997), ('n02109047', 'Great_Dane', 0.07607841), ('n02090379', 'redbone', 0.06192694)]
img_path = "/mnt/d/dog.drawing.1.jpg"
# [('n02088094', 'Afghan_hound', 0.3908979), ('n02091831', 'Saluki', 0.15265788), ('n02100735', 'English_setter', 0.016079867)]
img_path = "/mnt/d/dog.drawing.2.jpg"
# [('n01698640', 'American_alligator', 0.89597297), ('n01697457', 'African_crocodile', 0.062412985), ('n01665541', 'leatherback_turtle', 0.0003912723)]
img_path = "/mnt/d/aligator.1.png"
# [('n04350905', 'suit', 0.63127065), ('n04591157', 'Windsor_tie', 0.2877923), ('n02865351', 'bolo_tie', 0.015513565)]
img_path = "/mnt/d/tlj.jpg"
# [('n03710637', 'maillot', 0.56471676), ('n03710721', 'maillot', 0.083704114), ('n03255030', 'dumbbell', 0.05399074)]
img_path = "/mnt/d/superman.jpg"
# [('n03255030', 'dumbbell', 0.84658635), ('n02790996', 'barbell', 0.13791782), ('n04372370', 'switch', 0.0014434928)]
img_path = "/mnt/d/dumbbell.jpg"



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


