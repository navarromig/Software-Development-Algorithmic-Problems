#ifndef KMEANS_H
#define KMEANS_H

#include <vector>
#include <stdint.h>
#include <unordered_set>
#include <unordered_map>
#include <knn.h>


class Kmeans {
    private:
        std::unordered_map<std::string, int> &config;
        std::vector<std::vector<double> > &points;
        std::vector<std::vector<double> > centroids;

        int k;

        std::vector<int> assignments;
        std::vector<std::unordered_set<unsigned int> > clusters;

        KNN *knn;

        void assign_points();
        void assign_points_reverse();
        void assign_point(unsigned int point_index, int cluster);

        double get_average_distance_from_cluster(unsigned int point_index, int cluster_index);
    public:
        Kmeans(std::unordered_map<std::string, int> &config, std::vector<std::vector<double> > &points, int k);
        ~Kmeans();

        void run();
       
        std::vector<std::vector<double> > get_centroids();
        std::vector<std::unordered_set<unsigned int> > get_clusters();

        std::vector<double> silhouette();
};


#endif