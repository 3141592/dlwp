# Suppress warnings
import os, pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Force CPU use for keras.
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

print("11.4.3 The Transformer encoder")
import tensorflow as tf
from tensorflow import keras
batch_size = 16

train_ds = keras.utils.text_dataset_from_directory(
                "/root/src/data/aclImdb/train/", batch_size=batch_size)

val_ds = keras.utils.text_dataset_from_directory(
                "/root/src/data/aclImdb/val/", batch_size=batch_size)

test_ds = keras.utils.text_dataset_from_directory(
                "/root/src/data/aclImdb/test/", batch_size=batch_size)

text_only_train_ds = train_ds.map(lambda x, y: x)

print("Listing 11.12 Preparing integer sequence datasets")
from tensorflow.keras import layers

max_length = 600
max_tokens = 20000
text_vectorization = layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        # In order to keep a manageable input size, we'll truncate the inputs after the first 600 words.
        output_sequence_length=max_length,
)
text_vectorization.adapt(text_only_train_ds)

int_train_ds = train_ds.map(
                lambda x, y: (text_vectorization(x), y),
                num_parallel_calls=tf.data.AUTOTUNE)
int_val_ds = val_ds.map(
                lambda x, y: (text_vectorization(x), y),
                num_parallel_calls=tf.data.AUTOTUNE)
int_test_ds = test_ds.map(
                lambda x, y: (text_vectorization(x), y),
                num_parallel_calls=tf.data.AUTOTUNE)

print("Listing 11.21 Transformer encoder implemented as a subclassed Layer")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        # Size of input token vectors
        self.embed_dim = embed_dim
        # Size of tthe inner dense layer
        self.dense_dim = dense_dim
        # Number of attention heads
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
                [layers.Dense(dense_dim, activation="relu"),
                 layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    # Computation goes in call().
    def call(self, inputs, mask=None):
        # The mask that will be generated by tthe Embedding layer will be 2D,
        # but the attention layer expects to be 3D or 4D, so we expand its rank.
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(
                inputs,
                inputs,
                attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    # Implement serialization so we can save the model.
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config

print("Listing 11.22 Using the Transformer encoder for text classification")
vocab_size = 20000
embed_dim = 256
num_heads = 2
dense_dim = 32

inputs = keras.Input(shape=(None,), dtype="int64")
x = layers.Embedding(vocab_size, embed_dim)(inputs)
x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
# Since TransformerEncoder returns full sequences, we need to reduce each
# sequence to a single vector for classification via a global pooling layer.
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop",
        loss="binary_crossentropy",
        metrics=["accuracy"])
model.summary()

print("Listing 11.23 Training and evaluating the Transformer encoder based model")
callbacks = [
        keras.callbacks.ModelCheckpoint("transformer_encoder.keras",
        save_best_only=True)
]
model.fit(int_train_ds,
        validation_data=int_val_ds,
        epochs=20,
        callbacks=callbacks)
model = keras.models.load_model(
        "transformer_encoder.keras",
        # Provide the custom TransformerEncoder class to the model-loading process
        custom_objects={"TransformerEncoder": TransformerEncoder})
print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")



