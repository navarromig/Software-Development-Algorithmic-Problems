# python3 reduse.py -d input.dat -q query.dat -od reduced_input.dat -oq reduced_query.dat

import sys
from autoencoder import Autoencoder
from hyperparameter_tuning import hyperparameter_tuning
from file_io import read_mnist_data, write_mnist_data
import tensorflow.keras as keras


data_file = ''
query_file = ''
output_dataset_file = ''
output_query_file = ''

args = sys.argv
for i in range(1, len(args), 2):
    if args[i] == '-d':
        data_file = args[i+1]
    elif args[i] == '-q':
        query_file = args[i+1]
    elif args[i] == '-od':
        output_dataset_file = args[i+1]
    elif args[i] == '-oq':
        output_query_file = args[i+1]


if data_file == '' or query_file == '' or output_dataset_file == '' or output_query_file == '':
    print('Wrong arguments')    
    sys.exit(2)

x_train = read_mnist_data(data_file)
x_test = read_mnist_data(query_file)
x_train = x_train.reshape(x_train.shape+(1,)) / 255.0
x_test = x_test.reshape(x_test.shape+(1,)) / 255.0

# best_params = hyperparameter_tuning(x_train)
best_params = {'dropout_rate': 0.337382193446262, 'learning_rate': 3.739447945709745e-05}

#retraining the Autoencoder using the best parameters
best_dropout_rate = best_params['dropout_rate']
best_learning_rate = best_params['learning_rate']

autoencoder = Autoencoder(150, best_dropout_rate)
optimizer = keras.optimizers.Adam(learning_rate=best_learning_rate)
autoencoder.compile(optimizer=optimizer, loss='MSE')
autoencoder.fit(x_train, x_train, epochs=5, shuffle=True, validation_split=0.1, batch_size=64)

#evaluation of the retrained model
loss = autoencoder.evaluate(x_test, x_test, verbose=0)
print('Test loss:', loss)

red_x_train = autoencoder.encode(x_train)
red_x_test = autoencoder.encode(x_test)

red_x_train = red_x_train*255.0
red_x_test = red_x_test*255.0

red_x_train = red_x_train.astype(int)
red_x_test = red_x_test.astype(int)

write_mnist_data(red_x_train, output_dataset_file)
write_mnist_data(red_x_test, output_query_file)