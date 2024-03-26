#include <utils.h>
#include <knn.h>
#include "mrng.h"
#include "gnns.h"

#include <cstdlib>
#include <iostream>
#include <time.h> 
#include <fstream>
#include <chrono>
#include <string.h>
#include <numeric>
#include <algorithm>


int main(int argc, char** argv){

    srand(time(NULL));

    std::string input_file = "";
    std::string reduced_input_file = "";
    std::string query_file = "";
    std::string reduced_query_file = "";
    std::string output_file = "";

    // Default parameters
    int k = 50;
    int l = 20;
    int N = 1;
    int R = 1;
    int E = 30;

    int m = 0;

    // Parsing arguments
    for (int i = 1; i < argc; i+=2) {
        if (strcmp(argv[i], "-d") == 0) {
            input_file = argv[i + 1];
        } else if (strcmp(argv[i], "-rd") == 0) {
            reduced_input_file = argv[i + 1];
        } else if (strcmp(argv[i], "-q") == 0) {
            query_file = argv[i + 1];
        } else if (strcmp(argv[i], "-rq") == 0) {
            reduced_query_file = argv[i + 1];
        } else if (strcmp(argv[i], "-o") == 0) {
            output_file = argv[i + 1];
        } else if (strcmp(argv[i], "-k") == 0) {
            k = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-l") == 0) {
            l = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-N") == 0) {
            N = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-R") == 0) {
            R = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-E") == 0) {
            E = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-m") == 0) {
            m = atoi(argv[i + 1]);
        }
        else {
            std::cerr << "Invalid argument: " << argv[i] << std::endl;
            return -1;
        }
    }

    if (m <=0 || m > 3) {
        std::cerr << "Invalid method argument" << std::endl;
        return -1;
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
    std::vector<std::vector<double> > vectors;
    if (reduced_input_file != "") {
        vectors = readfile(reduced_input_file,1000);
    }
    else{
        vectors = readfile(input_file,1000);

    }

    KNN *graph;
    if (m == 1) {
        graph = new GNNS(vectors, k, E, R);
    } else if (m == 2){
        graph = new MRNG(vectors, l);
    }
    else {
        graph = new KNN(vectors);
    }

    KNN *base;
    std::vector<std::vector<double> > vectors_2;
    if (reduced_input_file != "") {
        vectors_2 = readfile(input_file,1000);
        base = new KNN(vectors_2);
    }else{
        base = new KNN(vectors);
    }

    do {

        std::vector<std::vector<double> > queries;
        if (reduced_query_file != "") {
            queries = readfile(reduced_query_file,100);
        }
        else{
            queries = readfile(query_file,100);
        }
        // Metrics
        std::vector<double> epsilon;
        std::vector<double> time_approx;
        std::vector<double> time_true;

        for (unsigned int i = 0; i < queries.size(); i++) {
            output << "Query " << i + 1 << std::endl;
            std::vector<double> &q = queries[i];

            auto start = std::chrono::high_resolution_clock::now();
            std::vector<unsigned int> knn_base = base->findKNN(N, q);
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            
            start = std::chrono::high_resolution_clock::now();
            std::vector<unsigned int> knn_approx =  graph->findKNN(N, q);
            stop = std::chrono::high_resolution_clock::now();
            auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    
            // Print results
            for(unsigned int j = 0; j < knn_approx.size(); j++) {
                double dist_approx = euclidean_distance(q, vectors[knn_approx[j]]);
                double dist_true = euclidean_distance(q, vectors[knn_base[j]]);

                output << "Nearest neighbor-" << j + 1 << ": " << knn_approx[j] << std::endl;
                output << "distanceApproximate: " << dist_approx << std::endl;
                output << "distanceTrue: " << dist_true << std::endl;
            }


            // Calculate epsilon
            double actual_distance = euclidean_distance(q, vectors[knn_base[0]]);
            double approx_distance = euclidean_distance(q, vectors[knn_approx[0]]);

            if (actual_distance == 0) {
                epsilon.push_back(approx_distance);
            } else {
                epsilon.push_back(approx_distance / actual_distance);
            }

            time_true.push_back(duration1.count());
            time_approx.push_back(duration2.count());
            
           
        }

        // Calculate metrics

        double mean_epsilon = std::accumulate(epsilon.begin(), epsilon.end(), 0.0) / epsilon.size();


        double mean_approx_time = std::accumulate(time_approx.begin(), time_approx.end(), 0.0) / time_approx.size();
        double mean_true_time = std::accumulate(time_true.begin(), time_true.end(), 0.0) / time_true.size();

        // Print metrics
        output << "Mean AF: " << mean_epsilon << std::endl;
        output << "tAverageApproximate: " << mean_approx_time << std::endl;
        output << "tAverageTrue: " << mean_true_time << std::endl;

        // Print metrics to stdout as a markdown row

        std::cout << "| " <<  mean_epsilon << " | " << mean_approx_time << " | " << mean_true_time << " |" << std::endl;
        std::cout << "Enter next query file or enter \"quit\" to exit: ";
        std::cin >> query_file;

    } while (query_file != "quit");


    return 0;
}
