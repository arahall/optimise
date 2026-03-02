#include <iostream>

#include <vector>
#include <functional>
#include <limits>
#include <numeric>
#include <algorithm>
#include <stdexcept>

#include "lbfgsb.h"

#define EIGEN_NO_DEBUG
//#define EIGEN_DONT_PARALLELIZE
#define EIGEN_VECTORIZE
#include "matrix_ops.h"

namespace
{
	inline double clamp(double x, double low, double hi)
	{
		return std::max(low, std::min(hi, x));
	}
	double get_optimality(const Eigen::VectorXd& x, const Eigen::VectorXd& g,
						  const Eigen::VectorXd& l, const Eigen::VectorXd& u)
	{
		size_t n = x.size();
		double max_element = 0.0;

		for (int i = 0; i < n; ++i)
		{
			double projected_value = std::min(std::max(l[i], x[i] - g[i]), u[i]);
			double difference = projected_value - x[i];
			max_element = std::max(max_element, std::abs(difference));
		}

		return max_element;  // Return the std::maximumstd::absolute value
	}

	double find_alpha(const Eigen::VectorXd& l, const Eigen::VectorXd& u,
					  const Eigen::VectorXd& xc, const Eigen::VectorXd& du,
					  const std::vector<size_t>& free_vars_idx)
	{
		double alpha_star = 1.0;
		size_t n = free_vars_idx.size();
		for (size_t i = 0; i < n; ++i)
		{
			size_t idx = free_vars_idx[i];
			if (du[i] > 0)
			{
				alpha_star = std::min(alpha_star, (u[idx] - xc[idx]) / du[i]);
			}
			else
			{
				alpha_star = std::min(alpha_star, (l[idx] - xc[idx]) / du[i]);
			}
		}
		return alpha_star;
	}
	void get_break_points(const Eigen::VectorXd& x, const Eigen::VectorXd& g,
						  const Eigen::VectorXd& l, const Eigen::VectorXd& u,
						  Eigen::VectorXd& t, Eigen::VectorXd& d)
	{
		// returns the break point vector and the search direction
		size_t n = x.size();
		if (t.size() != n || d.size() != n)
		{
			throw std::runtime_error("get_break_points - t and d must be the same size as x");
		}
		constexpr double realmax = std::numeric_limits<double>::max();
		constexpr double eps = std::numeric_limits<double>::epsilon();

		for (int i = 0; i < n; ++i)
		{
			if (g[i] < 0.0)
			{
				t[i] = (x[i] - u[i]) / g[i];
			}
			else if (g[i] > 0.)
			{
				t[i] = (x[i] - l[i]) / g[i];
			}
			else
			{
				t[i] = realmax;
			}
			d[i] = t[i] < eps ? 0.0 : -g[i];
		}
	}
	std::pair<Eigen::VectorXd, Eigen::VectorXd> get_cauchy_point(const Eigen::VectorXd& x, const Eigen::VectorXd& g,
											   const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
											   double theta, const Eigen::MatrixXd& w, const Eigen::MatrixXd& m)
	{
		size_t n = x.size();
		Eigen::VectorXd tt(n), d(n);
		get_break_points(x, g, lb, ub, tt, d);
		//
		std::vector<size_t> indices = matrix_ops::argsort(tt);
		Eigen::VectorXd xc = x;

		Eigen::VectorXd p = w.transpose() * d;
		Eigen::VectorXd c = Eigen::VectorXd::Zero(w.cols());
		double fp = -d.dot(d);
		double fpp = -theta * fp - p.dot(m * p);
		double fpp0 = -theta * fp;
		double dt_min = -fp / fpp;
		double t_old = 0.0;
		// examine the rest of the segments
		double epsilon_fpp0 = std::numeric_limits<double>::epsilon() * fpp0;
		Eigen::VectorXd mc = Eigen::VectorXd::Zero(w.cols());
		int k = 0;
		for (int i = 0; i < n; ++i)
		{
			int b = indices[i];
			double t = tt[b];
			double dt = t - t_old;
			if (dt_min <= dt)
			{
				k = i;
				break;
			}
			if (d[b] > 0.0)
			{
				xc[b] = ub[b];
			}
			else if (d[b] < 0.0)
			{
				xc[b] = lb[b];
			}
			double zb = xc[b] - x[b];
			Eigen::VectorXd mp = m * p;
			mc += mp * dt;
			c += p * dt;
			double gb = g[b];
			Eigen::VectorXd wb = w.row(b).transpose();
			Eigen::VectorXd mwb = m * wb;
			fp += dt * fpp + gb * gb + theta * gb * zb - gb * wb.dot(mc);  // m * c computed fresh
			fpp -= theta * gb * gb + 2.0 * gb * mwb.dot(p) + gb * gb * mwb.dot(wb);
			fpp = std::max(epsilon_fpp0, fpp);
			p += wb * gb;
			d[b] = 0.0;
			dt_min = -fp / fpp;
			t_old = t;
		}
		// perform final updates
		dt_min = std::max(dt_min, 0.0);
		t_old += dt_min;
		for (size_t j = k; j < n; ++j)
		{
			size_t idx = indices[j];
			xc[idx] += t_old * d[idx];
		}

		//c += p * dt_min;
		matrix_ops::add_scale(c, p, dt_min, c);
		return { xc, c };
	}

	bool subspace_minimisation(const Eigen::VectorXd& x, const Eigen::VectorXd& g, const Eigen::VectorXd& l, const Eigen::VectorXd& u,
							   const Eigen::VectorXd& xc, const Eigen::VectorXd& c, double theta, Eigen::MatrixXd& w, const Eigen::MatrixXd& m,
							  Eigen::VectorXd& xbar)
	{
		size_t n = x.size();
		std::vector<size_t> free_vars_index;

		for (size_t i = 0; i < n; ++i)
		{
			if (xc[i] > l[i] && xc[i] < u[i])
			{
				free_vars_index.push_back(i);
			}
		}
		size_t num_free_vars = free_vars_index.size();
		if (num_free_vars == 0)
		{
			xbar = xc;
			return false;
		}
		Eigen::MatrixXd wz = w(free_vars_index, Eigen::all);  // Directly extracting rows using index list

		// compute the reduced gradient of mk restricted to free variables
		// rr = g + theta * (xc - x) - w *(m*c)
		Eigen::VectorXd rr = g + (xc - x) * theta - w * (m * c);
		Eigen::VectorXd r(num_free_vars);
		for (int i = 0; i < num_free_vars; ++i)
		{
			r[i] = rr[free_vars_index[i]];
		}
		// form intermediate variables

		double one_over_theta = 1.0 / theta;
		Eigen::MatrixXd wz_T = wz.transpose();
		Eigen::VectorXd v = m * wz_T * r;
		Eigen::MatrixXd wz_T_wz = wz_T * wz;
		Eigen::MatrixXd big_n = Eigen::MatrixXd::Identity(wz_T_wz.rows(), wz_T_wz.cols());
		big_n.noalias() -= m * wz_T_wz * one_over_theta;  // Avoid aliasing
		v = big_n.colPivHouseholderQr().solve(v);
		Eigen::VectorXd du = -one_over_theta * r - one_over_theta * one_over_theta * wz * v;

		// find alpha star
		double alpha_star = find_alpha(l, u, xc, du, free_vars_index);

		// compute the subspace std::minimisation
		xbar = xc;
		for (size_t i = 0; i < num_free_vars; ++i)
		{
			size_t idx = free_vars_index[i];
			xbar[idx] += alpha_star * du[i];
		}
		for (int i = 0; i < n; ++i)
		{
			xbar[i] = clamp(xbar[i], l[i], u[i]);
		}
		return true;
	}

	double alpha_zoom(const std::function<double(const Eigen::VectorXd&)>& func,
					  const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& gradient,
					  const Eigen::VectorXd& x0,
					  double f0, const Eigen::VectorXd& g0, const Eigen::VectorXd& p,
					  double alpha_lo, double alpha_hi,
					  int max_iters, double c1, double c2)
	{
		size_t n = x0.size();
		Eigen::VectorXd x(n), x_lo(n);

		double dphi0 = g0.dot(p);

		for (int i = 0; i < max_iters; ++i)
		{
			double alpha_i = 0.5 * (alpha_lo + alpha_hi);
			matrix_ops::add_scale(x0, p, alpha_i, x);
			double f_i = func(x);
			matrix_ops::add_scale(x0, p, alpha_lo, x_lo);
			double f_lo = func(x_lo);

			if ((f_i > f0 + c1 * alpha_i * dphi0) || (f_i >= f_lo))
			{
				alpha_hi = alpha_i;
			}
			else
			{
				double dphi = gradient(x).dot(p);
				if (std::abs(dphi) <= -c2 * dphi0)
				{
					return alpha_i;
				}
				if (dphi * (alpha_hi - alpha_lo) >= 0)
				{
					alpha_hi = alpha_lo;
				}
				alpha_lo = alpha_i;
			}
		}
		return 0.5 * (alpha_hi + alpha_lo);
	}
	double strong_wolfe(const std::function<double(const Eigen::VectorXd&)>& func,
						const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& gradient,
						const Eigen::VectorXd& x0, double f0, const Eigen::VectorXd& g0,
						const Eigen::VectorXd& p, int max_iters, double c1, double c2, double alpha_max)
	{
		// compute line search satisfying strong Wolfe conditions
		double f_im1 = f0, alpha_im1 = 0.0, alpha_i = 1.0;
		double dphi0 = g0.dot(p);
		int n = x0.size();
		Eigen::VectorXd x(n);

		for (int iter = 0; iter < max_iters; ++iter)
		{
			matrix_ops::add_scale(x0, p, alpha_i, x);
			double f_i = func(x);
			if ((f_i > f0 + c1 * dphi0) || (iter > 1 && f_i >= f_im1))
			{
				return alpha_zoom(func, gradient, x0, f0, g0, p, alpha_im1, alpha_i, max_iters, c1, c2);
			}
			Eigen::VectorXd g_i = gradient(x);
			double dphi = g_i.dot(p);
			if (std::abs(dphi) <= -c2 * dphi0)
			{
				return alpha_i;
			}
			if (dphi >= 0.0)
			{
				return alpha_zoom(func, gradient, x0, f0, g0, p, alpha_i, alpha_im1, max_iters, c1, c2);
			}
			// update
			alpha_im1 = alpha_i;
			f_im1 = f_i;
			alpha_i += 0.8 * (alpha_max - alpha_i);
		}
		return alpha_i;
	}
	Eigen::MatrixXd hessian(const Eigen::MatrixXd& s, const Eigen::MatrixXd& y, double theta)
	{
		Eigen::MatrixXd sT = s.transpose();

		// Compute the matrix a and its lower triangular and diagonal parts
		Eigen::MatrixXd a = sT * y;
		Eigen::MatrixXd l = a.triangularView<Eigen::StrictlyLower>();
		Eigen::MatrixXd d = -1 * a.diagonal().asDiagonal();

		// Preallocate the final matrix mm
		Eigen::MatrixXd mm(d.rows() + l.rows(), d.cols() + l.rows());

		// Fill the top part of mm (d and l.transpose())
		mm.topLeftCorner(d.rows(), d.cols()) = d;
		mm.topRightCorner(d.rows(), l.rows()) = l.transpose();

		// Fill the bottom part of mm (l and theta * s.transpose() * s)
		mm.bottomLeftCorner(l.rows(), l.cols()) = l;
		mm.bottomRightCorner(l.rows(), s.cols()).noalias() = theta * sT * s;

		// Compute the inverse of the full matrix mm
		return mm.partialPivLu().inverse();
	}
}
bool LBFGSB::optimize(const std::function<double(const Eigen::VectorXd&)> &func,
					  const std::function<Eigen::VectorXd(const Eigen::VectorXd&)> &gradient,
					  Eigen::VectorXd& x, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
					  int max_history, int max_iter, int ln_srch_maxiter, 
					  double tol, double c1, double c2, double alpha_max,
					  double eps_factor, bool debug)
{
	// func - the function to be std::minimised
	// gradient - the gradient of the function to be std::minimosed
	// x - the solution 
	// max_history - number of corrections used in the limited memeory matrix
	// max_history < 3 not recommended, large m not recommend
	// 3 <= m < 20 is the recommended range for max_history
	//
	if (debug)
	{
		std::cout << "max_history: " << max_history << " max_iter: " << max_iter << " ln_srch_maxiter: " << ln_srch_maxiter <<
			" tol: " << tol << " c1: " << c1 << " c2: " << c2 << " alpha_max: " << alpha_max << std::endl;
	}
	size_t n = x.size(); // the problem dimension
	double tol_f = std::numeric_limits<double>::epsilon() * eps_factor;
	// check that the bounds are well specified
	for (unsigned i = 0; i < n; ++i)
	{
		if (lb[i] >= ub[i])
		{
			throw std::runtime_error("LBFGSB::optimise - lower bound must be less than upper boound");
		}
	}
	
	Eigen::MatrixXd w = Eigen::MatrixXd::Zero(n, 1), m = Eigen::MatrixXd::Zero(1, 1);
	Eigen::MatrixXd y_history, s_history;
	Eigen::VectorXd xbar(n);
	
	constexpr double eps = std::numeric_limits<double>::epsilon();
	double f = func(x);
	Eigen::VectorXd g = gradient(x);
	if (g.size() != n)
	{
		throw std::runtime_error("LBFGSB::optimise - length of gradient must be the same as the problem dimension");
	}
	double theta = 1.0;
	for (int iter = 0; iter < max_iter; ++iter)
	{
		double opt = get_optimality(x, g, lb, ub);
		if (debug)
		{
			std::cout << "optimality = " << opt << " func = " << f << "\n";
			for (const auto& x_ : x)
			{
				std::cout << x_ << " ";
			}
			std::cout << "\n";
		}
		if (opt < tol)
		{
			if (debug)
			{
				std::cout << "converged in " << iter << " iterations\n";
			}
			return true;
		}
		Eigen::VectorXd x_old(x);
		Eigen::VectorXd g_old(g);

		// compute new search directon
		Eigen::VectorXd xc, c;
		std::tie(xc, c) = get_cauchy_point(x, g, lb, ub, theta, w, m);
		bool flag = subspace_minimisation(x, g, lb, ub, xc, c, theta, w, m, xbar);
		Eigen::VectorXd dx = xbar - x;
		double alpha = flag ? std::min(1.0, strong_wolfe(func, gradient, x, f, g, dx, ln_srch_maxiter, c1, c2, alpha_max))
			: 1.0;
		x += alpha * dx;
		double f_new = func(x);
		if (debug)
		{
			std::cout << "f_new: " << f_new << " f: " << f << "\n";
		}
		double f_tol_check =std::abs(f_new - f) / std::max(std::max(std::abs(f), 1.0), std::abs(f_new));
		if (f_tol_check <= tol_f)
		{
			if (debug)
			{
				std::cout << "converged in " << iter << "iterations due to function tolerance: " <<
					f_tol_check << "tolf: " << tol_f << "\n";
			}
			return true;
		}
		f = f_new;
		g = gradient(x);
		dx = x - x_old;
		Eigen::VectorXd dg = g - g_old;
		double curv = dx.dot(dg);
		if (curv >= eps)
		{
			if (y_history.cols() == max_history)
			{
				matrix_ops::remove_end_column(y_history);
				matrix_ops::remove_end_column(s_history);
			}
			matrix_ops::add_end_column(y_history, dg);
			matrix_ops::add_end_column(s_history, dx);
			theta = dg.dot(dg) / dg.dot(dx);
			w = Eigen::MatrixXd(y_history.rows(), y_history.cols() + s_history.cols());
			w << y_history, theta* s_history;
			m = hessian(s_history, y_history, theta);
		}
		if (debug && curv < eps)
		{
			std::cout << "optimise - negative curvature detected. Hessian update skipped\n";
		}
	}
	return false;
}
