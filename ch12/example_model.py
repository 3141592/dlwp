import tensorflow as tf

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: CheckPoints
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath = "CDX_Best.h5", monitor='val_accuracy', save_best_only = True, verbose=1)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Class / Definition
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class MyLSTMLayer( tf.keras.layers.LSTM ):
    def __init__(self, units, return_sequences, return_state):
        super(MyLSTMLayer, self).__init__( units, return_sequences=True, return_state=False )
        self.num_units = units

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
        shape=[int(input_shape[-1]),
        self.num_units])

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Variables
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
start = 3
limit = 12
delta = 3
sample = tf.range( start, limit, delta )
sample = tf.cast( sample, dtype=tf.float32 )
sample = tf.constant( sample, shape=( 1, 1, 3 ) )
layer = MyLSTMLayer( 3, True, False )

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Model Initialize
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1, 3)),
    tf.keras.layers.Reshape( (1, 3) ),
    layer,
])

model.summary()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
DataSet
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
dataset = tf.data.Dataset.from_tensor_slices((tf.constant( sample, shape=( 1, 1, 1, 3 ) ), tf.constant([0], shape=(1, 1, 1, 1))))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Optimizer
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
optimizer = tf.keras.optimizers.Nadam(
    learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
    name='Nadam'
)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Loss Fn
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""                               
lossfn = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False,
    reduction=tf.keras.losses.Reduction.AUTO,
    name='sparse_categorical_crossentropy'
)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Model Summary
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
model.compile(optimizer=optimizer, loss=lossfn, metrics=['accuracy'])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Training
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
history = model.fit( dataset, validation_data=dataset, batch_size=100, epochs=50, callbacks= [checkpointer] )

print( sample )
print( model.predict(sample) )
