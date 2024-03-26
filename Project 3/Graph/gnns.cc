#include "gnns.h"

#include <queue>
#include <unordered_set>
#include <stdlib.h>
#include <limits>
#include <iostream>


#include "../LSH/lsh.h"

GNNS::GNNS(std::vector<std::vector<double> > &vectors, int k, int E, int R, double (*distance_function)(std::vector<double> &, std::vector<double> &)) : KNN(vectors), graph(vectors.size()), k(k), E(E), R(R) {
    // Initialize LSH
    LSH lsh(vectors,8,20);

    // Find the k nearest neighbors for each vector
    for (unsigned int i = 0; i < vectors.size(); i++) {
        std::vector<unsigned int> knn = lsh.findKNN(k + 1, vectors[i], distance_function);

        for (unsigned int j = 0; j < knn.size(); j++) {
            if (knn[j] != i) {
                graph[i].push_back(knn[j]);
            }
           
        }
    }
}

std::vector<unsigned int> GNNS::findKNN(int k, std::vector<double> &q, double (*distance_function)(std::vector<double> &, std::vector<double> &)) {
    std::vector<unsigned int> knn;
    
    std::unordered_set<unsigned int> S;

    for (int i = 0; i < R; i++) {
        int vec_id = rand() % graph.size(); // Randomly select the starting vector

        S.insert(vec_id);

        // Find the next vector on the path
        for (int j = 0; j < t && vec_id != -1; j++) {
            std::vector<unsigned int> &neighbors = graph[vec_id];
            double min_dist = std::numeric_limits<double>::max();
            int min_id = -1;
            for (unsigned int l = 0; l < (unsigned int)E && l < neighbors.size(); l++) {
                S.insert(graph[vec_id][l]);

                double dist = distance_function(vectors[graph[vec_id][l]], q);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_id = graph[vec_id][l];
                }
            }
            vec_id = min_id;
        }
    }

    // Find the KNN
    std::priority_queue<std::pair<double, unsigned int> > pq;

    for (auto it = S.begin(); it != S.end(); it++) {
        pq.push(std::make_pair(distance_function(vectors[*it], q), *it));
    }

    for (int i = 0; i < k && !pq.empty(); i++) {
        knn.push_back(pq.top().second);
        pq.pop();
    }

    return knn;
}
