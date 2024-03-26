#ifndef GNNS_H
#define GNNS_H

#include <vector>

#include <knn.h>

class GNNS : public KNN {
    private:
        std::vector<std::vector<unsigned int> > graph;

        int k;
        int E;
        int R;

        int t = 20;

    public:
        GNNS(std::vector<std::vector<double> > &vectors, int k, int E, int R, double (*distance_function)(std::vector<double> &, std::vector<double> &) = euclidean_distance);

        virtual std::vector<unsigned int> findKNN(int k, std::vector<double> &q, double (*distance_function)(std::vector<double> &, std::vector<double> &) = euclidean_distance);
};





#endif // GNNS_H