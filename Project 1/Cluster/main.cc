#include <iostream>

#include "kmeans.h"
#include <utils.h>

#include <string>
#include <fstream>
#include <unordered_map>
#include <random>
#include <time.h>
#include <string.h>
#include <chrono>


void read_config(std::string filename,int &k, std::unordered_map<std::string, int> &config);


int main(int argc, char** argv) {

    srand(time(NULL));

    std::string input_file = "";
    std::string conf_file = "";
    std::string output_file = "";

    // Default parameters
    bool complete = false;

    std::string method;

    // Parsing arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0) {
            input_file = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "-c") == 0) {
            conf_file = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "-o") == 0) {
            output_file = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "-complete") == 0) {
            complete = true;
        } else if (strcmp(argv[i], "-m") == 0) {
            method = argv[i+1];
            i++;
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

    if (conf_file == "") {
        std::cout << "Enter configuration file: ";
        std::cin >> conf_file;
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

    // Reading vectors

    std::vector<std::vector<double> > points = readfile(input_file);

    int k;
    std::unordered_map<std::string, int> config;

    read_config(conf_file, k, config);

    for (auto it = config.begin(); it != config.end(); it++){
        std::cout << it->first << ": " << it->second << std::endl;
    }

    if (method == "Classic"){
        config["type"] = 0;
    }
    else if (method == "LSH"){
        config["type"] = 1;
    }
    else if (method == "Hypercube"){
        config["type"] = 2;
    }
    else{
        std::cerr << "Error there is no such a method" << std::endl;
        exit(1);
    }
    
    Kmeans kmeans(config, points, k);
    auto start = std::chrono::high_resolution_clock::now();
    kmeans.run();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);    

    output << "Algorithm: ";

    if (method == "Classic"){
        output << "Lloyd's";
    }
    else if (method == "LSH"){
        output << "Range Search LSH";
    }
    else if (method == "Hypercube"){
        output << "Range Search Hypercube";
    }

    output << std::endl;

    std::vector<std::vector<double> > centroids = kmeans.get_centroids();

    for (int i = 0; i < k; i++){
        output << "CLUSTER-" << i+1 << " {size: " << centroids[i].size() << ", centroid: ";

        for (unsigned int j = 0; j < centroids[i].size(); j++){
            output << centroids[i][j] << " ";
        }
        output << "}" << std::endl;
        
    }

    output << "clustering_time: " << duration.count()/1000.0 << std::endl;

    std::vector<double> silhouette = kmeans.silhouette();
    output << "Silhouette: [";
    for (unsigned int i = 0; i < silhouette.size(); i++){
        output << silhouette[i] << " ";
    }
    output << "]" << std::endl;
    

    if (complete) {
        std::vector<std::unordered_set<unsigned int> > clusters = kmeans.get_clusters();
        for (int i = 0; i < k; i++) {
            output << "CLUSTER-" << i + 1 << " {";

            for (unsigned int j = 0; j <centroids[i].size(); j++) {
                output << centroids[i][j] << " ";
            }

            for (auto it = clusters[i].begin(); it != clusters[i].end(); it++) {
                output << *it << " ";
            }


            output << "}" << std::endl;

        }
    }

    output.close();
    return 0;
}

void read_config(std::string filename,int &k, std::unordered_map<std::string, int> &config){
    config["number_of_clusters"] = 10;
    config["number_of_hash_functions"] = 3;
    config["number_of_hash_tables"] = 4;
    config["max_number_M_hypercube"] = 10;
    config["number_of_hypercube_dimensions"] = 3;
    config["number_of_probes"] = 2;

    
    std::ifstream file(filename);
    std::string line;

    while (getline(file, line)){
        auto pos = line.find_first_of(":");
        std::string key = line.substr(0, pos);
        std::string value_str = line.substr(pos+1);
        int value;

        sscanf(value_str.c_str(), "%d", &value);

        if (key == "number_of_clusters"){
            k = value;
        } else {
            config[key] = value;
        }
    }
    file.close();
}