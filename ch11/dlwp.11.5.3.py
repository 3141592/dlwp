# Suppress warnings
import os, pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Force CPU use for keras.
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

print("11.5.3 Sequence-to-sequence learning with Transformer")
print("Listing 11.33 The TransformerDecoder")
import tensorflow as tf
from tensorflow.keras import layers

class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim= embed_dim)
        self.attention_2 = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim= embed_dim)
        self.dense_proj = keras.Sequential(
                [layers.Dense(dense_dim, activation="relu"),
                 layers.Dense(embed_dim),]
                )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        # This attribute ensures that the layer will propogate its input mask to its outputs.
        # If you pass a mask to a layer that doesn't implement compute_mask() and that doesn't
        # expose this supports_masking attribute, that's an error.
        self.supports_masking = True

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
            })
        return config

print("Listing 11.34 TransformerDecoder method that generates a causal mask")
def get_causal_attention_mask(self, inputs):
    input_shape = tf.shape(inputs)
    batch_size, sequence_length = input_shape[0], input_shape[1]
    i = tf.range(sequence_length)[:, tf.newaxis]
    j = tf.range(sequence_length)
    mask = tf.cast(i >= j, dtype="int32")
    mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
    mult = tf.concat(
            [tf.expand_dims(batch_size, -1),
             tf.constant([1, 1], dtype=tf.int32)], axis=0)
    return tf.tile(mask, mult)






