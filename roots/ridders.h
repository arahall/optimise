#ifndef _RIDDERS_H
#define _RIDDERS_H

#include <functional>
using namespace std;

namespace RIDDERS
{
	// root finding function
	double root(function<double(double)> func, double xl, double xh, int max_iter = 100, double atol = 1e-6,
				double rtol = 1e-4, double ftol = 1e-8);
};

#endif