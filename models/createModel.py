import tensorflow as tf
from models.TransformerEncoder import TransformerEncoder
from models.Time2Vec import Time2Vector


def create_model(seq_len, d_k, d_v, n_heads, ff_dim):
    '''Initialize time and transformer layers'''
    time_embedding = Time2Vector(seq_len)
    attn_layer1 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
    attn_layer2 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
    attn_layer3 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)

    '''Construct model'''
    in_seq = tf.keras.Input(shape=(seq_len, 16))
    x = time_embedding(in_seq)
    x = tf.keras.layers.Concatenate(axis=-1)([in_seq, x])
    x = attn_layer1((x, x, x))
    x = attn_layer2((x, x, x))
    x = attn_layer3((x, x, x))
    x = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_first')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    out = tf.keras.layers.Dense(1, activation='linear')(x)

    model = tf.keras.models.Model(inputs=in_seq, outputs=out)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer='adam', metrics=['mae', 'mape', 'binary_accuracy'])
    return model
