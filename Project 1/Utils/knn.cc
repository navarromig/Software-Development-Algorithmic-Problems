#include "knn.h"
#include <queue>

KNN::KNN(std::vector<std::vector<double> > &vectors) : vectors(vectors) {}

std::vector<unsigned int > KNN::findKNN(int k, std::vector<double> &q, double (*distance_function)(std::vector<double> &, std::vector<double> &)){

	std::vector<unsigned int > knn(k);
	std::priority_queue<std::pair<double, unsigned int> > pq;

	for (unsigned int i = 0; i < vectors.size(); i++){
		double distance = distance_function(q, vectors[i]);
		pq.push(std::make_pair(distance, i));
		if (pq.size() > (unsigned int) k) {
			pq.pop();
		}
			
	}

	for (int i = k - 1; i >= 0; i--) {
        knn[i] = pq.top().second;
        pq.pop();
    }

	return knn; 
}

std::vector<unsigned int > KNN::range_search(double radius, std::vector<double> &q, double (*distance_function)(std::vector<double> &, std::vector<double> &)) {
	std::vector<unsigned int > vectors_in_range;

	for (unsigned int i = 0; i < vectors.size(); i++){
		double distance = distance_function(q, vectors[i]);
		if (distance <= radius){
			vectors_in_range.push_back(i);
		}
	}

	return vectors_in_range; 
}