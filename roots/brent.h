#ifndef _BRENT_H
#define _BRENT_H

#include <functional>
using namespace std;

namespace BRENT
{
	double root(function<double(double)> func, double a, double b, int max_iter = 100, double atol = 1.0e-6,
				double ftol = 1e-8);
};
#endif