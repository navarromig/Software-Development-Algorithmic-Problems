#ifndef HYPERCUBE_H
#define HYPERCUBE_H

#include <knn.h>
#include <h_function.h>

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

class Hypercube : public KNN {
    private:
        int M;
        int probes;
        int dim;

        std::vector<std::unordered_map<unsigned int, double> > f_functions;
        std::vector<H_function *> h_functions;

        std::unordered_map<uint64_t, std::unordered_set<unsigned int> > hypercube_nodes;

        double f(int i, unsigned int val);

        uint64_t project(std::vector<double> &v);

        std::vector<uint64_t> get_vectors_upto_hamming_distance(uint64_t v, int hamming_distance);
    public:
        Hypercube(std::vector<std::vector<double> > &vectors, int M, int probes, int dim);

        ~Hypercube();
        
        std::vector<unsigned int> findKNN(int k, std::vector<double> &q, double (*distance_function)(std::vector<double> &, std::vector<double> &) = euclidean_distance);
        
        std::vector<unsigned int> range_search(double radius, std::vector<double> &q, double (*distance_function)(std::vector<double> &, std::vector<double> &) = euclidean_distance);
};

#endif