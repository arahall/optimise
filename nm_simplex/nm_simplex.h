#ifndef _NM_SIMPLEX_H
#define _NM_SIMPLEX_H

#include <Eigen/Dense>
#include <functional>

namespace NelderMeadSimplex
{
	bool optimize(const std::function<double(const Eigen::VectorXd&)> &func, Eigen::VectorXd &p,
				  double alpha = 1.0, double gamma = 2.0, double rho = 0.5, double sigma = 0.5,
				  double tol = 1.0e-4, int max_iter = 200, int max_restarts = 5);
};
#endif
