#ifndef _NM_SIMPLEX_H
#define _NM_SIMPLEX_H

#include <Eigen/Dense>
#include <functional>

namespace NelderMeadSimplex
{
	bool optimise(std::function<double(const Eigen::VectorXd&)> func, Eigen::VectorXd &p,
				  double tol = 1.0e-4, int max_iter = 200);
};
#endif
