#ifndef H_FUNCTION_H
#define H_FUNCTION_H

#include <vector>
#include <cstdint>

class H_function {
	private:
		std::vector<double> p;
		double t;
		double W;
	public:
		H_function(int dim, int W);
		unsigned int get_value(std::vector<double> &v);
};

#endif