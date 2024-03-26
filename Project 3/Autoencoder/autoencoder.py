import tensorflow.keras as keras
from tensorflow.keras import Model

import numpy as np

class Autoencoder(Model):
    def __init__(self, latent_dim, dropout_rate=0.0):
        super(Autoencoder, self).__init__()
        self.encoder = keras.Sequential([
            keras.layers.Input(shape=(28, 28, 1)),
            keras.layers.Conv2D(16, (3, 3), activation= 'elu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Conv2D(32, (3, 3), activation='elu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(latent_dim, activation='sigmoid')
        ])
        self.decoder = keras.Sequential([
            keras.layers.Input(shape=(latent_dim,)),
            keras.layers.Dense(11*11*16, activation='elu'),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Reshape((11, 11, 16)),
            keras.layers.Conv2DTranspose(16, (3, 3), activation='elu'),
            keras.layers.UpSampling2D((2, 2)),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Conv2DTranspose(1, (3, 3), activation='sigmoid')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        encoded = []
        for i in range(0, len(x), 64):
            encoded = encoded + self.encoder(x[i:i+64]).numpy().tolist()
        encoded = np.array(encoded)

        return encoded