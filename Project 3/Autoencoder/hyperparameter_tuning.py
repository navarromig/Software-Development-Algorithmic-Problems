import optuna
import tensorflow as tf

from autoencoder import Autoencoder

    
def objective(trial, x_train):
    
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)

    
    autoencoder = Autoencoder(dropout_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    autoencoder.compile(optimizer= optimizer, loss='MSE')
    
    history = autoencoder.fit(x_train, x_train, epochs=5, shuffle=True, validation_split=0.1, batch_size=64)
    
    return history.history['val_loss'][-1]

def hyperparameter_tuning(x_train):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, x_train), n_trials=5)
    
    return study.best_params