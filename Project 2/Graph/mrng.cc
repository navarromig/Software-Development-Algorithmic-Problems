#include "mrng.h"
#include <math.h>
#include <iostream>
#include <limits>
#include <queue>
#include <set>

#include <algorithm>

MRNG::MRNG(std::vector<std::vector<double> > &vectors, int L, double (*distance_function)(std::vector<double> &, std::vector<double> &)) : KNN(vectors), L(L) {
    
    outgoing_edges.resize(vectors.size());

    for (int p = 0; p < vectors.size(); p++) {


        std::vector<unsigned int> Rp;  // Rp = S − {p}
        std::priority_queue<std::pair<double, unsigned int> > pq;
        std::vector<double> distances;      //All the distances from p to others nodes

        for (int i = 0; i < vectors.size(); i++) {
            if (i != p) {

                double distance = distance_function(vectors[i], vectors[p]);
                pq.push(std::make_pair(distance, i));
                distances.push_back(distance);
            }
        }

        while( !pq.empty() ) {
            Rp.push_back(pq.top().second);
            pq.pop();
        };



        // Find nodes in Rp that have the minimum distance to p
        double min_distance = *min_element(distances.begin(), distances.end());
        
		// Initialize Lp = {x ∈ Rp | δ(x, p) = min{δ(y, p) : y ∈ Rp}}
        std::vector<unsigned int > Lp;
        for (int i = 0; i < Rp.size(); ++i) {  	
            if (distances[i] == min_distance) {
                Lp.push_back(Rp[i]);
            }
        }

        bool condition = true;
        int r;
        for (r=0; r < Rp.size(); r++) {

            if (std::find(Lp.begin(), Lp.end(), Rp[r]) == Lp.end()) {   //Vector Rp[r] it's not in Lp

                for (int t=0; t < Lp.size(); t++) {
                    //Checking if pr is the longest edge in triangle (prt) 
                    if (distance_function(vectors[p], vectors[r]) >= distance_function(vectors[r], vectors[t]) && distance_function(vectors[p], vectors[r]) >= distance_function(vectors[p], vectors[t])) {
                        condition = false;
                        break;
                    }
                }
                if (!condition) {
                    break;
                }
            }
        }
        if (condition) {
            Lp.push_back(Rp[r]);
        }
        outgoing_edges[p].resize(Lp.size());
        for (int i = 0; i < Lp.size(); ++i) {
            outgoing_edges[p][i] = Lp[i];
        }
    }

    //Calculating the centroid, because it's necesarry for finding the navigate node
    std::vector<double> centroid(vectors[0].size(), 0.0);

	for (int i=0; i<vectors.size(); i++) {
		for (int j = 0; j < vectors[0].size(); j++) {
			centroid[j] += vectors[i][j];
		}
	}

	//Calculating the mean for each dimension
	for (int i = 0; i < vectors[0].size(); i++) {
		centroid[i] /= vectors.size();
	}

	std::vector<unsigned int> node = KNN::findKNN(1, centroid);
    navigate_node = node[0];
}


std::vector<unsigned int>  MRNG::findKNN(int k, std::vector<double> &q, double (*distance_function)(std::vector<double> &, std::vector<double> &)){
	
	std::vector<unsigned int> knn;

    std::priority_queue<std::pair<double, unsigned int> >  R;

	std::priority_queue<std::pair<double, unsigned int> > pq;    //This pq will be modified during the algorithm, by inserting and deleting items. In the Rp only we will insert items.

    std::vector<bool> checked(vectors.size(), false);  //Bool vector to hold the checked vectors 

    std::vector<bool> into_R(vectors.size(), false);   //Bool vector to hold the id of the vectors into the R

	pq.push(std::make_pair(distance_function(vectors[navigate_node], q), navigate_node));
    R.push(std::make_pair(distance_function(vectors[navigate_node], q), navigate_node));
    into_R[navigate_node] = true;

	int i = 0;
    
	while(i < L && !pq.empty()){
        
        unsigned int p = pq.top().second;
        pq.pop();
		checked[p] = true;



		for(int z=0; z < outgoing_edges[p].size(); z++){    //For every neighbor N of p ∈ Outgoing_edges:
            unsigned int neighbor = outgoing_edges[p][z];
			if (!into_R[neighbor] && !checked[neighbor]) {     //N !∈ R  && N it's not checked
                into_R[neighbor] = true;
				pq.push(std::make_pair(distance_function(vectors[neighbor], q), neighbor));
                R.push(std::make_pair(distance_function(vectors[neighbor], q), neighbor));
				i++;
			}
		}
	}

	for (int i = 0; i < k && !R.empty(); i++) {
        knn.push_back(R.top().second);
        R.pop();
    }

    return knn;
};