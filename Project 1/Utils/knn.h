#ifndef BASE_H
#define BASE_H

#include "utils.h"
#include <vector>

class KNN{
	protected:
		std::vector<std::vector<double> > &vectors;

	public:
		KNN(std::vector<std::vector<double> >&vectors);
		virtual ~KNN() {};

		virtual std::vector<unsigned int> findKNN(int k, std::vector<double> &q, double (*distance_function)(std::vector<double> &, std::vector<double> &) = euclidean_distance);
		virtual std::vector<unsigned int> range_search(double radius, std::vector<double> &q, double (*distance_function)(std::vector<double> &, std::vector<double> &) = euclidean_distance);
};


#endif
