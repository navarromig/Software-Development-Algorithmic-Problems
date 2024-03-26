#include "hypercube.h"

#include <queue>
#include <utility>
#include <iostream>


Hypercube::Hypercube(std::vector<std::vector<double> > &vectors, int M, int probes, int dim) : KNN(vectors), M(M), probes(probes), 
                                                                 dim(dim), f_functions(dim) {
    // Initialize h functions
    for (int i = 0; i < dim; i++) {
        h_functions.push_back(new H_function(vectors[0].size(), 4000));
    }
    
    // Project vectors 
    for (unsigned int i = 0; i < vectors.size(); i++) {
        std::vector<double> &v = vectors[i];
        uint64_t projected_v = project(v);
        
        hypercube_nodes[projected_v].insert(i);
    }

}

Hypercube::~Hypercube() {
    for (unsigned int i = 0; i < h_functions.size(); i++) {
        delete h_functions[i];
    }
}

std::vector<unsigned int> Hypercube::findKNN(int k, std::vector<double> &q, double (*distance_function)(std::vector<double> &, std::vector<double> &)) {        
    std::vector<unsigned int > knn(k);

    uint64_t projected_q = project(q);

    std::vector<uint64_t> vectors_to_check = get_vectors_upto_hamming_distance(projected_q, probes);

    std::priority_queue<std::pair<double, unsigned int> > pq;

    int vectors_checked = 0;

    // Get the k nearest neighbors
    for (auto it = vectors_to_check.begin(); it != vectors_to_check.end() && vectors_checked < M; it++) {
        uint64_t v = *it;
        if (hypercube_nodes.find(v) != hypercube_nodes.end()) {
            std::unordered_set<unsigned int> &indexes = hypercube_nodes[v];
            for (auto it = indexes.begin(); it != indexes.end() && vectors_checked < M; it++) {
                unsigned int index = *it;
                double distance = distance_function(vectors[index], q);
                pq.push(std::make_pair(distance, index));

                if (pq.size() > (unsigned int) k) {
                    pq.pop();
                }
                vectors_checked++;
            }
        }
    }

    for (int i = pq.size() - 1; i >= 0; i--) {
        knn[i] = pq.top().second;
        pq.pop();
    }

    return knn;  
}

std::vector<unsigned int > Hypercube::range_search(double radius, std::vector<double> &q, double (*distance_function)(std::vector<double> &, std::vector<double> &)) {  
    std::vector<unsigned int > vectors_in_range;

    uint64_t projected_q = project(q);

    std::vector<uint64_t> vectors_to_check = get_vectors_upto_hamming_distance(projected_q, probes);

    int vectors_checked = 0;

    // Get the vectors inside the range
    for (auto it = vectors_to_check.begin(); it != vectors_to_check.end() && vectors_checked < M; it++) {
        uint64_t v = *it;
        if (hypercube_nodes.find(v) != hypercube_nodes.end()) {
            std::unordered_set<unsigned int> &indexes = hypercube_nodes[v];
            for (auto it = indexes.begin(); it != indexes.end() && vectors_checked < M; it++) {
                unsigned int index = *it;
                double distance = distance_function(vectors[index], q);
                if (distance <= radius) {
                    vectors_in_range.push_back(index);
                }
                vectors_checked++;
            }
        }
    }

    return vectors_in_range;  
}

double Hypercube::f(int i, unsigned int val) {
    if (f_functions[i].find(val) == f_functions[i].end()) {
        f_functions[i][val] = rand() % 2;
    }
    return f_functions[i][val];
}

uint64_t Hypercube::project(std::vector<double> &v) {
    // Project a vector to a vertex in the hypercube

    uint64_t projected_v = 0;
    for (int i = 0; i < dim; i++) {
        projected_v = projected_v << 1;
        unsigned int val = h_functions[i]->get_value(v);
        projected_v += f(i, val);
    }

    return projected_v;

}

std::vector<uint64_t> Hypercube::get_vectors_upto_hamming_distance(uint64_t v, int hamming_distance) {
    // Get all the vectors that are up to hamming_distance away from v
  
    std::vector<uint64_t> vectors;
   
    std::unordered_set<uint64_t> vectors_set;  // Used to avoid duplicates

    vectors.push_back(v);
    vectors_set.insert(v);

    if (probes == 1) {
        return vectors;
    }

    for (int i = 0; i < hamming_distance; i++) {
        std::unordered_set<uint64_t> new_vectors;

        // Find the new vectors of hamming distance i + 1
        for (auto it = vectors_set.begin(); it != vectors_set.end(); it++) {
            uint64_t v = *it;
            for (int j = 0; j < dim; j++) {
                uint64_t new_v = v ^ (1 << j);
                new_vectors.insert(new_v);
            }
        }

        // Insert the new vectors
        for (auto it = new_vectors.begin(); it != new_vectors.end(); it++) {
            uint64_t v = *it;
            if (vectors_set.find(v) == vectors_set.end()) {
                vectors.push_back(v);
                vectors_set.insert(v);
                if (vectors.size() == (unsigned int)probes) {
                    return vectors;
                }
            }

        }
    }

    return vectors;
}