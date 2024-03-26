#include "utils.h"

#include <fstream>
#include <iostream>
#include <random>

#include <fstream>
#include <iostream>
#include <random>
#include <vector>

unsigned int reverseInt(unsigned int i);

std::vector<std::vector<double> > readfile(std::string filename, unsigned int limit) {
    // Reads an MNIST dataset and returns a vector of vectors of doubles with the images

    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error opening file" << std::endl;
        exit(1);
    }

    std::vector<std::vector<double> > vectors;

    unsigned int magic_number;
    unsigned int number_of_images;
    unsigned int number_of_rows;
    unsigned int number_of_columns;

    // Read the header of the file

    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    file.read((char *)&number_of_images, sizeof(number_of_images));
    number_of_images = reverseInt(number_of_images);
    file.read((char *)&number_of_rows, sizeof(number_of_rows));
    number_of_rows = reverseInt(number_of_rows);
    file.read((char *)&number_of_columns, sizeof(number_of_columns));
    number_of_columns = reverseInt(number_of_columns);

    std::cout << "Reading file: " << filename << std::endl;

    std::cout << "Magic number: " << magic_number << std::endl;
    std::cout << "Number of images: " << number_of_images << std::endl;
    std::cout << "Number of rows: " << number_of_rows << std::endl;
    std::cout << "Number of columns: " << number_of_columns << std::endl;

    // Read the images

    if (limit != 0) {
        number_of_images = limit;
    }

    for (unsigned int i = 0; i < number_of_images; i++) {
        std::vector<double> image;
        for (unsigned int j = 0; j < number_of_rows * number_of_columns; j++) {
            uint8_t pixel;
            file >> pixel;
            image.push_back((double)pixel);
        }
        vectors.push_back(image);
    }

    std::cout << "Finished " << std::endl;

    file.close();

    return vectors;
}

unsigned int reverseInt(unsigned int i) {
    // Reverses the bytes of an integer

    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((unsigned int)c1 << 24) + ((unsigned int)c2 << 16) + ((unsigned int)c3 << 8) + c4;
}

std::vector<double> create_random_vector(int dim, int mean, int stddev, bool square_value) {
    std::vector<double> v(dim);
    static std::random_device rd;
    static std::default_random_engine generator(rd());
    std::normal_distribution<double> distribution(mean, stddev);
    for (int i = 0; i < dim; i++) {
        double val = distribution(generator);
        if (square_value) {
            v[i] = val * val;
        } else {
            v[i] = val;
        }
    }

    return v;
}

double euclidean_distance(std::vector<double> &v1, std::vector<double> &v2) {
    double sum = 0;
    for (unsigned int i = 0; i < v1.size(); i++) {
        sum += (v1[i] - v2[i]) * (v1[i] - v2[i]);
    }
    return sqrt(sum);
}
