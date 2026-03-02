#ifndef _LBFGSB_H
#define _LBFGSB_H

#include <functional>
#include <Eigen/Dense>

#define EIGEN_NO_DEBUG
#define EIGEN_VECTORIZE

namespace LBFGSB
{
	// optimise function
	bool optimize(const std::function<double(const Eigen::VectorXd&)> &func,
				  const std::function<Eigen::VectorXd(const Eigen::VectorXd&)> &grad,
				  Eigen::VectorXd& x, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
				  int max_history = 5, int max_iter = 100, int ln_srch_maxiter = 10, double tol = 1e-7,
				  double c1 = 1e-4, double c2 = 0.9, double alpha_max = 2.5, double eps_factor = 1e7, 
				  bool debug = false);
};
#endif // !_LBFGSB_H
