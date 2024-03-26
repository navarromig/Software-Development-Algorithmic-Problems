#include "h_function.h"
#include "utils.h"
#include <stdlib.h>

#include <iostream>


H_function::H_function(int dim, int W) : W(W) {
    p = create_random_vector(dim);
    t = ((double) rand()) / RAND_MAX * W;
}

unsigned int H_function::get_value(std::vector<double> &v) {
    double sum = 0;
    for (unsigned int i = 0; i < v.size(); i++) {
        sum += p[i] * v[i];
    }

    return  (unsigned int) ((sum + t) / W);
}