#ifndef MRNG_H
#define MRNG_H


#include <vector>
#include <knn.h>
#include <h_function.h>


#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>

class MRNG: public KNN {
	private:
		std::vector<std::vector<unsigned int> > outgoing_edges;

        int L;
		unsigned int navigate_node;

	public:

		MRNG(std::vector<std::vector<double> > &vectors, int I, double (*distance_function)(std::vector<double> &, std::vector<double> &) = euclidean_distance);
		std::vector<unsigned int> findKNN(int k, std::vector<double> &q, double (*distance_function)(std::vector<double> &, std::vector<double> &) = euclidean_distance);
};


#endif 