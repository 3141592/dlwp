# Suppress warnings
import os, pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Force CPU use for keras.
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

print("12.3.3 Nerual style transfer in Keras")

print("Listing 12.16 Getting the style and content images")
import tensorflow as tf
from tensorflow import keras

base_image_path = keras.utils.get_file("sf.jpg", origin="https://img-datasets.s3.amazonaws.com/sf.jpg")
style_reference_image_path = keras.utils.get_file("starry_might.jpg", origin="https://img-datasets.s3.amazonaws.com/starry_night.jpg")

original_width, original_height = keras.utils.load_img(base_image_path).size
img_height = 400
img_width = round(original_width * img_height / original_height)

print("Listing 12.17 Auxillary functions")
import numpy as np

# Util functions to open, resize, and format pictures into appropriate arrays
def preprocess_image(image_path):
    img = keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img = keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = keras.applications.vgg19.preprocess_input(img)
    return img

# Util function to convert a NumPy array into a valid image
def deprocess_image(img):
    img = img.reshape((img_height, img_width, 3))
    # Zero-centering by removing the mean pixel value from ImageNet.
    # This reverses a transformation done by vgg19.preprocess_input.
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    # Converts images from 'BGR' to 'RGB'.
    # This is also part of the reversal of vgg19.preprocess_input.
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype("uint8")
    return img

print("Listing 12.18 Using a pretrained VGG19 model to create a feature extractor")
# Build a VGG19 model loaded with pretrained ImageNet weights.
model = keras.applications.vgg19.VGG19(weights="imagenet", include_top=False)
model.summary()

outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
# Model that returns the activation values for every target layer (as a dict)
feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

print("Listing 12.19 Content loss")
def content_loss(base_img, combination_img):
    return tf.reduce_sum(tf.square(combination_img - base_img))

print("Listing 12.20 Style loss")
def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram

def style_loss(style_img, combination_img):
    S = gram_matrix(style_img)
    C = gram_matrix(combination_img)
    channels = 3
    size = img_height * img_width
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size **2))

print("Listing 12.21 Total variation loss")
def total_variation_loss(x):
    a = tf.square(
            x[:, : img_height - 1, : img_width -1, :] - x[:, 1:, : img_width - 1, :]
            )
    b = tf.square(
            x[:, : img_height - 1, : img_width -1, :] - x[:, : img_height - 1, 1:, :]
            )
    return tf.reduce_sum(tf.pow(a + b, 1.25))

print("Listing 12.22 Defining the final loss that you'll minimize")
# List of layers to use for the style loss
style_layer_names = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1"
        ]
# The layer to use for the content loff
content_layer_name = "block5_conv2"
# Contribution weight of the total variation loss
total_variation_weight = 1e-6
# Contribution weight of the style loss
style_weight = 1e-6
# Contribution weight of the content loss
content_weight = 2.5e-8

def compute_loss(combination_image, base_image, style_reference_image):
    input_tensor = tf.concat(
            [base_image, style_reference_image, combination_image], axis=0)
    features = feature_extractor(input_tensor)
    # Initialize the loss to 0.
    loss = tf.zeros(shape=())
    # Add the content loss.
    layer_features = features[content_layer_name]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + content_weight + content_loss(
            base_image_features, combination_features
            )
    # Add the style loss.
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        style_loss_value = style_loss(
                style_reference_features, combination_features)
        loss += (style_weight / len(style_layer_names)) + style_loss_value

    # Add the total variation loss
    loss += total_variation_weight * total_variation_loss(combination_image)
    return loss

print("Listing 12.23 Setting up the gradient-descent process")
import tensorflow as tf

# We make the training step fast by compiling it as a ttf.function.
@tf.function
def compute_loss_and_grads(combination_image, base_image, style_reference_image):
    with tf.GradientTape() as tape:
        loss = compute_loss(combination_image, base_image, style_reference_image)
        grads = tape.gradient(loss, combination_image)
        return loss, grads

optimizer = keras.optimizers.SGD(
        # We'll start with a learning rate of 100 and decrease by 4% every 100 steps.
        keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
            )
        )

base_image = preprocess_image(base_image_path)
style_reference_image = preprocess_image(style_reference_image_path)
# Use a Variable to store the combination image since we'll be updating it during training.
combination_image = tf.Variable(preprocess_image(base_image_path))

iterations = 4000
for i in range(1, iterations + 1):
    loss, grads = compute_loss_and_grads(
            combination_image, base_image, style_reference_image
            )
    # Update the combination image in a direction that reduces the style transfer loss.
    optimizer.apply_gradients([(grads, combination_image)])
    if i % 100 == 0:
        print(f"Iteration {i}: loss={loss:.2f}")
        img = deprocess_image(combination_image.numpy())
        fname = f"combination_image_at_iteration_{i}.png"
        # Save the combination image at regluar intervals.
        keras.utils.save_img(fname, img)

# See https://stackoverflow.com/a/40434284



