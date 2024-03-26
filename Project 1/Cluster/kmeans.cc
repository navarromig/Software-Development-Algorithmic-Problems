#include "kmeans.h"

#include <unordered_map>
#include <vector>
#include <limits>
#include <random>
#include <iostream>

#include <utils.h>

#include "../LSH/lsh.h"
#include "../Hypercube/hypercube.h"


Kmeans::Kmeans(std::unordered_map<std::string, int> &config, std::vector<std::vector<double> > &points, int k) : config(config), points(points), k(k), assignments(points.size()), clusters(k) {  
    
    //Choosing random point as first centroid
    int firstCentroidsIndex = rand() % points.size();
    centroids.push_back(points[firstCentroidsIndex]);

    std::unordered_set<int> indexesCentroids;

    indexesCentroids.insert(firstCentroidsIndex);

    for (int i = 1; i < k; i++){

        std::vector<double> D(points.size() - i, std::numeric_limits<double>::max());
        int index_D = 0;

        //A map to store the match of point-indexes to D-indexes
        std::unordered_map<unsigned int, unsigned int> index_pairs(points.size() - i);


        //Calculating D[j]
        for (unsigned int j = 0; j < points.size(); j++){
            if (indexesCentroids.find(j) == indexesCentroids.end()) {

                double dist;
                //For each choosen centroid
                for (int z = 0; z < i; z++){
                    dist = euclidean_distance(points[j], centroids[z]);
                    D[index_D] = std::min(D[index_D], dist);
                }
                index_pairs.insert(std::make_pair(index_D,j));
                index_D++;
            }
        }
        double max_d = *max(D.begin(), D.end());
        for (unsigned int j = 0; j < points.size() - i; j++){
            D[j] = D[j] / max_d;
        }

        //Calculating partialSum
        std::vector<double> partialSum(points.size() - i);
        partialSum[0] = D[0] * D[0];
        for (unsigned int r = 1; r < points.size() - i; r++){
            partialSum[r] = partialSum[r-1] + D[r] * D[r];
        }
        double lower_bound = 0;
        double upper_bound = partialSum[partialSum.size() - 1];
    
        std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
        std::default_random_engine re;
        double x = unif(re);
        partialSum[0] = 0;

        //Searching r
        auto it = std::lower_bound(partialSum.begin(), partialSum.end(), x);
        int r = it - partialSum.begin();
        

        int index = index_pairs[r];

        indexesCentroids.insert(index);
        centroids.push_back(points[index]);

        assignments[index] = i;
        clusters[i].insert(index);
    }

    for (unsigned int i = 0; i < points.size(); i++){
        if (indexesCentroids.find(i) == indexesCentroids.end()) {
            assignments[i] = -1;
        }
    }

    if (config["type"] == 1) {
        int k = config["number_of_vector_hash_functions"];
        int L = config["number_of_vector_hash_tables"];
        knn = new LSH(points, k, L);
    }
    else if (config["type"] == 2) {
        int k = config["number_of_hypercube_dimensions"];
        int M = config["max_number_M_hypercube"];
        int probes = config["number_of_probes"];
        knn = new Hypercube(points, M, probes, k);
    }
}

Kmeans::~Kmeans() {
    if (config["type"] == 1 || config["type"] == 2) {
        delete knn;
    }
}

void Kmeans::run() {
    if (config["type"] == 0) {
        assign_points();
    } else {
        assign_points_reverse();
    }
}

std::vector<std::vector<double> > Kmeans::get_centroids() {
    return centroids;
}

std::vector<std::unordered_set<unsigned int> > Kmeans::get_clusters() {
    return clusters;
}

void Kmeans::assign_points() {
    bool changed;
    int iterations = 0;

    do {
        std::cout << "Iteration: " << iterations << std::endl;
        changed = false;
        knn = new KNN(centroids);

        for (unsigned int i = 0; i < points.size();i++) {

            std::vector<unsigned int> neighbours = knn->findKNN(1, points[i]);
            int cluster = neighbours[0];

            if (assignments[i] != cluster) {
                assign_point(i, cluster);
                changed = true;
            }

        }

        delete knn;

        iterations++;
    } while (changed);

    std::cout << "Iterations: " << iterations << std::endl;
}

void Kmeans::assign_points_reverse() {

    bool changed;

    int iterations = 0;

    do {
        std::cout << "Iteration: " << iterations << std::endl;

        double R = 1e3;
        int cnt_assigned;

        changed = false;

        std::vector<bool> assigned(points.size());
        
        for (unsigned int i = 0;i < points.size();i++) {
            assigned[i] = false;
        }
        
        do {
            cnt_assigned = 0;
            std::unordered_map<unsigned int, std::unordered_set<int> > assignments_to_test;
            for (int i = 0;i < k;i++) {
                std::vector<unsigned int> neighbours = knn->range_search(R, centroids[i]);
                for (unsigned int j = 0;j < neighbours.size();j++) {
                    if (!assigned[neighbours[j]]) {
                        assignments_to_test[neighbours[j]].insert(i);
                    }
                }
            }



            for (std::unordered_map<unsigned int, std::unordered_set<int> >::iterator it = assignments_to_test.begin(); it != assignments_to_test.end(); it++) {
                unsigned int point_index = it->first;
                std::unordered_set<int> &possible_assignments = it->second;

                int chosen_centroid = -1;
                double min_distance = std::numeric_limits<double>::max();

                for (std::unordered_set<int>::iterator it2 = possible_assignments.begin(); it2 != possible_assignments.end(); it2++) {
                    int centroid_index = *it2;
                    double distance = euclidean_distance(points[point_index], centroids[centroid_index]);
                    if (distance < min_distance) {
                        min_distance = distance;
                        chosen_centroid = centroid_index;
                    }
                }

                if (assignments[point_index] != chosen_centroid) {
                    assign_point(point_index, chosen_centroid);
                    changed = true;
                }

                cnt_assigned++;
                assigned[point_index] = true;
            }


            R *= 2;

            std::cout << "number of assigned points: " << cnt_assigned << std::endl;
            fflush(stdout);
        } while (cnt_assigned > 10 || R < 1e5);

        //Unassigned points. Comparing their distances to all centroids
        for (unsigned int i = 0;i < points.size();i++) {
            if (!assigned[i]) {
                int chosen_centroid = 0;
                int min_distance = euclidean_distance(points[i], centroids[0]);
                for (int j = 0;j < k;j++) {
                    int distance = euclidean_distance(points[i], centroids[j]);
                    if (distance < min_distance) {
                        min_distance = distance;
                        chosen_centroid = j;
                    }
                }
                
                if (assignments[i] != chosen_centroid) {
                    assign_point(i, chosen_centroid);
                    changed = true;
                }
            }
        }

        iterations++;

    } while (changed);
}


void Kmeans::assign_point(unsigned int point_index, int cluster) {
    int prev_cluster = assignments[point_index];

    for (unsigned int i = 0; i < centroids[cluster].size(); i++) {
        centroids[cluster][i] = (centroids[cluster][i] * clusters[cluster].size() + points[point_index][i]) / (clusters[cluster].size() + 1);
    }

    if (prev_cluster != -1) {
        for (unsigned int i = 0; i < centroids[prev_cluster].size(); i++) {
            if (clusters[prev_cluster].size() == 1) {
                centroids[prev_cluster][i] = 0;
            } else {
                centroids[prev_cluster][i] = (centroids[prev_cluster][i] * clusters[prev_cluster].size() - points[point_index][i]) / (clusters[prev_cluster].size() - 1);
            }
        }
        clusters[prev_cluster].erase(point_index);
    }

    clusters[cluster].insert(point_index);
    assignments[point_index] = cluster;    
}


std::vector<double> Kmeans::silhouette() {
    std::vector<double> silhouette(points.size());
    double sum = 0;
    for (unsigned  i = 0; i < points.size(); i++) {
        int cluster_index = assignments[i];

        int second_cluster_index;

        if (cluster_index == 0) {
            second_cluster_index = 1;
        } else {
            second_cluster_index = 0;
        }

        double min_distance = euclidean_distance(points[i], centroids[second_cluster_index]);

        for (int j = second_cluster_index + 1; j < k; j++) {
            if (j != cluster_index) {
                double distance = euclidean_distance(points[i], centroids[j]);
                if (distance < min_distance) {
                    min_distance = distance;
                    second_cluster_index = j;
                }
            }
        }

        double a = get_average_distance_from_cluster(i, cluster_index);
        double b = get_average_distance_from_cluster(i, second_cluster_index);

        silhouette[i] = (b - a) / std::max(a, b);

        sum += silhouette[i];

        std::cout << "Silhouette calculation: " << i << "/" << points.size() << std::endl;
    }

    std::vector<double> result(k + 1);

    for (int i = 0; i < k; i++) {
        double sum = 0;
        for (std::unordered_set<unsigned int>::iterator it = clusters[i].begin(); it != clusters[i].end(); it++) {
            unsigned int index = *it;
            sum += silhouette[index];
        }
        
        result[i] = sum / clusters[i].size();

        std::cout << "cluster: " << i << std::endl;
        std::cout << "sum: " << sum << std::endl;
        std::cout << "size: " << clusters[i].size() << std::endl;
        std::cout << "result: " << result[i] << std::endl << std::endl;
    }

    result[k] = sum / points.size();

    return result;
}

double Kmeans::get_average_distance_from_cluster(unsigned int point_index, int cluster_index) {
    double sum = 0;
    for (std::unordered_set<unsigned int>::iterator it = clusters[cluster_index].begin(); it != clusters[cluster_index].end(); it++) {
        unsigned int index = *it; 
        sum += euclidean_distance(points[point_index], points[index]);
    }

    return sum / (clusters[cluster_index].size() - 1);
}
