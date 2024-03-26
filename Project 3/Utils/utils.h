#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>

std::vector<std::vector<double> > readfile(std::string filename, unsigned int limit = 0);
std::vector<double> create_random_vector(int dim, int mean = 0, int stddev = 1,bool square_value = false);

double euclidean_distance(std::vector<double> &v1, std::vector<double> &v2);

#endif