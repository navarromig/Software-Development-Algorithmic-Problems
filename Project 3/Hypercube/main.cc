#include "hypercube.h"
#include <knn.h>
#include <utils.h>

#include <chrono>
#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <algorithm>
#include <numeric>


int main(int argc, char** argv) {
    srand(time(NULL));

    std::string input_file = "";
    std::string query_file = "";
    std::string output_file = "";

    // Default parameters
    int k = 14;
    int M = 10;
    int probes = 2;
    int N = 1;
    double R = 10000;

    // Parsing arguments
    for (int i = 1; i < argc; i+=2) {
        if (strcmp(argv[i], "-d") == 0) {
            input_file = argv[i + 1];
        } else if (strcmp(argv[i], "-q") == 0) {
            query_file = argv[i + 1];
        } else if (strcmp(argv[i], "-k") == 0) {
            k = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-M") == 0) {
            M = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-probes") == 0) {
            probes = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-o") == 0) {
            output_file = argv[i + 1];
        } else if (strcmp(argv[i], "-N") == 0) {
            N = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-R") == 0) {
            R = atof(argv[i + 1]);
        }
        else {
            std::cout << "Invalid argument: " << argv[i] << std::endl;
            return -1;
        }
    }

    // Get missing arguments

    if (input_file == "") {
        std::cout << "Enter input file: ";
        std::cin >> input_file;
    }

    if (query_file == "") {
        std::cout << "Enter query file: ";
        std::cin >> query_file;
    }

    if (output_file == "") {
        std::cout << "Enter output file: ";
        std::cin >> output_file;
    }

    std::ofstream output(output_file);

    if (!output.is_open()) {
        std::cerr << "Error opening output file" << std::endl;
        exit(1);
    }


    // Get vectors and initialize the two objects
    std::vector<std::vector<double> > vectors = readfile(input_file,1000);

    Hypercube hypercube = Hypercube(vectors, M, probes, k);
    KNN base(vectors);

    do {
        std::vector<std::vector<double> > queries = readfile(query_file,100);
        // Metrics
        std::vector<double> epsilon;
        int true_nn = 0;

        std::vector<double> time_approx;
        std::vector<double> time_true;

        for (unsigned int i = 0; i < queries.size(); i++) {
            output << "Query " << i + 1 << std::endl;
            std::vector<double> &q = queries[i];

            auto start = std::chrono::high_resolution_clock::now();
            std::vector<unsigned int> knn_hypercube = hypercube.findKNN(N, q);
            auto stop = std::chrono::high_resolution_clock::now();

            auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

            start = std::chrono::high_resolution_clock::now();
            std::vector<unsigned int> knn_base = base.findKNN(N, q);
            stop = std::chrono::high_resolution_clock::now();
            auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

            // Print results
            for(unsigned int j = 0; j < knn_hypercube.size(); j++) {
                output << "Nearest neighbor-" << j + 1 << ": " << knn_hypercube[j] << std::endl;
                output << "distanceLsh: " << euclidean_distance(q, vectors[knn_hypercube[j]]) << std::endl;
                output << "distanceTrue: " << euclidean_distance(q, vectors[knn_base[j]]) << std::endl;
            }

            output << "tLSH: " << duration1.count()/1000.0 << std::endl;
            output << "tTrue: " << duration2.count()/1000.0 << std::endl;

            
            if (euclidean_distance(q, vectors[knn_hypercube[0]]) == euclidean_distance(q, vectors[knn_base[0]])) {
                true_nn++;
            }

            output << "R-near neighbors:" << std::endl;
            std::vector<unsigned int> range_hypercube = hypercube.range_search(R, q);


            // Calculate epsilon
            double actual_distance = euclidean_distance(q, vectors[knn_base[0]]);
            double approx_distance = euclidean_distance(q, vectors[knn_hypercube[0]]);
            if (actual_distance == 0) {
                epsilon.push_back((approx_distance - actual_distance));
            } else {
                epsilon.push_back((approx_distance - actual_distance) / actual_distance);
            }

            time_approx.push_back(duration1.count());
            time_true.push_back(duration2.count());

            // Range search
            for (unsigned int j = 0; j < range_hypercube.size(); j++) {
                output << range_hypercube[j] << std::endl;
            }
           
        }

        // Calculate metrics
        double mean_epsilon = std::accumulate(epsilon.begin(), epsilon.end(), 0.0) / epsilon.size();


        double mean_approx_time = std::accumulate(time_approx.begin(), time_approx.end(), 0.0) / time_approx.size();
        double mean_true_time = std::accumulate(time_true.begin(), time_true.end(), 0.0) / time_true.size();

        // Print metrics


        std::cout << "| " << mean_epsilon << " | " << mean_approx_time << " | " << mean_true_time << " |" << std::endl;

        std::cout << "Enter next query file or enter \"quit\" to exit: ";
        std::cin >> query_file;

    } while (query_file != "quit");

    output.close();

    return 0;
}
