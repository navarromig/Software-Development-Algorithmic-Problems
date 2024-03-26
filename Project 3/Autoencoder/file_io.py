import numpy as np

def read_mnist_data(data_file):
    with open(data_file, 'rb') as f:
        _ = int.from_bytes(f.read(4), 'big')
        number_of_images = int.from_bytes(f.read(4), 'big')
        number_of_rows = int.from_bytes(f.read(4), 'big')
        number_of_columns = int.from_bytes(f.read(4), 'big')
        
        data = []
        for i in range(number_of_images):
            image = []
            for j in range(number_of_rows):
                row = []
                for k in range(number_of_columns):
                    row.append(int.from_bytes(f.read(1), 'big'))
                image.append(row)
            data.append(image)
        

        data = np.array(data)
    return data

def write_mnist_data(data, output_file):
    with open(output_file, 'wb') as f:
        f.write(int(0).to_bytes(4, 'big'))
        f.write(len(data).to_bytes(4, 'big'))
        f.write(len(data[0]).to_bytes(4, 'big'))
        f.write(int(1).to_bytes(4, 'big'))

        for image in data:
            for pixel in image:
                f.write(int(pixel).to_bytes(1, 'big'))
