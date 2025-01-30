#ifndef _LBFGSB_H
#define _LBFGSB_H

#include <functional>
#include "matrix_ops.h"

using namespace std;

#if defined(USING_EIGEN)
#define EIGEN_NO_DEBUG
//#define EIGEN_DONT_PARALLELIZE
#define EIGEN_VECTORIZE
#endif

namespace LBFGSB
{
	// optimise function
	bool optimize(function<double(const VectorXd&)> func,
				  function<VectorXd(const VectorXd&)> grad,
				  VectorXd& x, const VectorXd& lb, const VectorXd& ub,
				  int max_history = 5, int max_iter = 100, int ln_srch_maxiter = 10, double tol = 1e-7,
				  double c1 = 1e-4, double c2 = 0.9, double alpha_max = 2.5, double eps_factor = 1e7, 
				  bool debug = false);
};
#endif // !_LBFGSB_H
