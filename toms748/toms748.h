#ifndef _TOMS_748_H
#define _TOMS_748_H

#include <functional>
using namespace std;

namespace TOMS748
{
	// root finding function
	double root(function<double(double)> func, double a, double b, int neps, int max_iter=100, double tol=1e-6);
};
#endif
