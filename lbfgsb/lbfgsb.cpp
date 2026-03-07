#include <iostream>

#include <vector>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <algorithm>
#include <stdexcept>

#include "lbfgsb.h"

#define EIGEN_NO_DEBUG
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
			if (dt_min <= dt) {
				k = i;
				break;
			}
			if (d[b] > 0.0) xc[b] = ub[b];
			else if (d[b] < 0.0) xc[b] = lb[b];

			double zb = xc[b] - x[b];
			c += p * dt;                          // update c first
			double gb = g[b];
			Eigen::VectorXd wb = w.row(b).transpose();
			Eigen::VectorXd mwb = m * wb;         // M*w_b, reused twice
			Eigen::VectorXd mc = m * c;           // M*c, fresh each iteration
			fp += dt * fpp + gb * gb + theta * gb * zb - gb * wb.dot(mc);
			fpp -= theta * gb * gb + 2.0 * gb * mwb.dot(p) + gb * gb * mwb.dot(wb);
			fpp = std::max(epsilon_fpp0, fpp);
			p += wb * gb;
			d[b] = 0.0;
			dt_min = -fp / fpp;
			t_old = t;
		}		// perform final updates
		dt_min = std::max(dt_min, 0.0);
		t_old += dt_min;
		for (size_t j = k; j < n; ++j)
		{
			size_t idx = indices[j];
			xc[idx] += t_old * d[idx];
		}

		c += p * dt_min;
		return { xc, c };
	}

	bool subspace_minimisation(const Eigen::VectorXd& x, const Eigen::VectorXd& g,
							   const Eigen::VectorXd& l, const Eigen::VectorXd& u,
							   const Eigen::VectorXd& xc, const Eigen::VectorXd& c,
							   double theta, const Eigen::MatrixXd& w, const Eigen::MatrixXd& M,
							   Eigen::VectorXd& xbar)
	{
		size_t n = x.size();
		std::vector<size_t> free_vars_index;
		free_vars_index.reserve(n);
		for (size_t i = 0; i < n; ++i)
		{
			if (xc[i] > l[i] && xc[i] < u[i])
			{
				free_vars_index.push_back(i);
			}
		}
		size_t nf = free_vars_index.size();
		if (nf == 0)
		{
			xbar = xc;
			return false;
		}

		// compute mc once, reuse in rr
		Eigen::VectorXd mc = M * c;

		// reduced gradient
		Eigen::VectorXd rr = g + (xc - x) * theta - w * mc;

		// extract free variable rows — wz built separately from r
		Eigen::MatrixXd wz = w(free_vars_index, Eigen::all);
		Eigen::VectorXd r(nf);
		for (size_t i = 0; i < nf; ++i)
		{
			r[i] = rr[free_vars_index[i]];
		}

		// form intermediate variables — unchanged from original
		double one_over_theta = 1.0 / theta;
		Eigen::MatrixXd wz_T = wz.transpose();
		Eigen::VectorXd v = M * (wz_T * r);
		Eigen::MatrixXd wz_T_wz = wz_T * wz;
		Eigen::MatrixXd big_n = Eigen::MatrixXd::Identity(wz_T_wz.rows(), wz_T_wz.cols());
		big_n.noalias() -= M * wz_T_wz * one_over_theta;
		v = big_n.partialPivLu().solve(v);  // faster than colPivHouseholderQr

		Eigen::VectorXd du = -one_over_theta * r - one_over_theta * one_over_theta * wz * v;

		// find alpha star and update xbar
		double alpha_star = find_alpha(l, u, xc, du, free_vars_index);
		xbar = xc;
		for (size_t i = 0; i < nf; ++i)
		{
			size_t idx = free_vars_index[i];
			xbar[idx] += alpha_star * du[i];
		}
		for (size_t i = 0; i < n; ++i)
		{
			xbar[i] = std::clamp(xbar[i], l[i], u[i]);
		}
		return true;
	}

	// More-Thuente line search
	double zoom(const std::function<double(const Eigen::VectorXd&)>& func,
				const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& gradient,
				const Eigen::VectorXd& x0, const Eigen::VectorXd& p, double alpha_lo, double alpha_hi,
				double f0, double g0, double f_lo, double g_lo, double f_hi, double g_hi, 
				double c1, double c2, int max_iter, Eigen::VectorXd& grad_out)
	{
		auto cubic_minimizer = [](double a, double fa, double ga, double b, double fb, double gb) -> 
			std::optional<double>
			{
				double d1 = ga + gb - 3 * (fa - fb) / (a - b);
				double d2 = d1 * d1 - ga * gb;
				if (d2 < 0.0)
				{
					return std::nullopt;
				}
				d2 = std::sqrt(d2);
				return b > a ? b - (b - a) * ((gb + d2 - d1) / (gb - ga + 2 * d2)) : 
					b + (a - b) * ((gb + d2 - d1) / (gb - ga + 2 * d2));
			};
		auto quadratic_minimizer = [](double a, double fa, double ga, double b, double fb) -> std::optional<double>
			{
				double denom = 2 * (fa - fb + (b - a) * ga);
				if (denom > 0.0 || denom < 0.0)
				{
					return a - ga * (b - a) * (b - a) / denom;
				}
				return std::nullopt;
			};

		double alpha;

		for (int iter = 0; iter < max_iter; ++iter)
		{
			// try cubic interpolation
			auto alpha_new = cubic_minimizer(alpha_lo, f_lo, g_lo, alpha_hi, f_hi, g_hi);
			if (!alpha_new)
			{
				// fall back
				alpha_new = quadratic_minimizer(alpha_lo, f_lo, g_lo, alpha_hi, f_hi);
				if (!alpha_new)
				{
					alpha = 0.5 * (alpha_lo + alpha_hi);
				}
				else
				{
					alpha = *alpha_new;
				}
			}
			else
			{
				alpha = *alpha_new;
			}
			double lo = std::min(alpha_lo, alpha_hi);
			double hi = std::max(alpha_lo, alpha_hi);
			double width = hi - lo;
			alpha = std::clamp(alpha, lo + 0.1 * width, hi - 0.1 * width);
			//
			Eigen::VectorXd x_new = x0 + alpha * p;
			double f_new = func(x_new);
			grad_out = gradient(x_new);
			double g_new = grad_out.dot(p);
			if ((f_new > f0 + c1 * alpha * g0) || (f_new >= f_lo))
			{
				alpha_hi = alpha; f_hi = f_new; g_hi = g_new;
			}
			else
			{
				if (std::abs(g_new) <= -c2 * g0)
				{
					return alpha;
				}
				if (g_new * (alpha_hi - alpha_lo) >= 0)
				{
					alpha_hi = alpha_lo; f_hi = f_lo; g_hi = g_lo;
				}
				alpha_lo = alpha;
				f_lo = f_new;
				g_lo = g_new;
			}
		}
		return alpha;
	}

	double more_thuente(const std::function<double(const Eigen::VectorXd&)>& func,
						const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& gradient,
						const Eigen::VectorXd& x0, double f0, const Eigen::VectorXd& grad_0, double alpha_init,
						const Eigen::VectorXd& p, int max_iters, double c1, double c2, double alpha_max, 
						Eigen::VectorXd& grad_out)
	{
		double alpha0 = 0, alpha1 = alpha_init, g0 = grad_0.dot(p);
		if (g0 > 0.0)
		{
			throw std::runtime_error("search direction is not steepest descent\n");
		}
		double f_prev = f0, g_prev = g0;
		for (int iter = 0; iter < max_iters; ++iter)
		{
			Eigen::VectorXd x_new = x0 + alpha1 * p;
			double f_new = func(x_new);
			grad_out = gradient(x_new);
			double g_new = grad_out.dot(p);
			if ((f_new > f0 + c1 * alpha1 * g0) || (iter > 0 && f_new >= f_prev))
			{
				return zoom(func, gradient, x0, p, alpha0, alpha1, f0, g0, f_prev, g_prev, f_new, g_new, 
							c1, c2, max_iters, grad_out);
			}
			if (std::abs(g_new) <= -c2 * g0)
			{
				return alpha1;
			}
			if (g_new >= 0.0)
			{
				return zoom(func, gradient, x0, p, alpha1, alpha0, f0, g0, f_new, g_new, f_prev, g_prev, 
							c1, c2, max_iters, grad_out);
			}
			alpha0 = alpha1;
			f_prev = f_new;
			g_prev = g_new;
			alpha1 = std::min(2. * alpha1, alpha_max);
		}
		return alpha1;
	}

	Eigen::MatrixXd hessian(const Eigen::MatrixXd& s, const Eigen::MatrixXd& y, double theta)
	{
		const int m = s.cols();
		Eigen::MatrixXd a = s.transpose() * y;

		// Preallocate the final matrix mm
		Eigen::MatrixXd mm(2*m, 2*m);

		// Fill the top part of mm (d and l.transpose())
		mm.topLeftCorner(m, m).setZero();
		mm.topLeftCorner(m, m).diagonal() = -a.diagonal();
		// top right 
		mm.topRightCorner(m, m) = a.triangularView<Eigen::StrictlyLower>().transpose();
		// Bottom-left: strictly lower tri of a
		mm.bottomLeftCorner(m, m) = a.triangularView<Eigen::StrictlyLower>();
		// Bottom-right: theta * s^T * s
		mm.bottomRightCorner(m, m).noalias() = theta * s.transpose() * s;

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
	
	double f = func(x);
	Eigen::VectorXd g = gradient(x);
	if (g.size() != n)
	{
		throw std::runtime_error("LBFGSB::optimise - length of gradient must be the same as the problem dimension");
	}
	double alpha_init = 1.0;
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
		if (iter > 0)
		{
			w = Eigen::MatrixXd(y_history.rows(), y_history.cols() + s_history.cols());
			w << y_history, theta* s_history;
			m = hessian(s_history, y_history, theta);
		}
		std::tie(xc, c) = get_cauchy_point(x, g, lb, ub, theta, w, m);
		bool flag = subspace_minimisation(x, g, lb, ub, xc, c, theta, w, m, xbar);

		Eigen::VectorXd dx = xbar - x;
		double alpha = flag ? std::min(1.0, more_thuente(func, gradient, x, f, g, alpha_init, 
									   dx, ln_srch_maxiter, c1, c2, alpha_max, g)) : 1.0;
		x += alpha * dx;
		double f_new = func(x);
		if (debug)
		{
			std::cout << "f_new: " << f_new << " f: " << f << "\n";
		}
		double f_tol_check = std::abs(f_new - f) / std::max(std::max(std::abs(f), 1.0), std::abs(f_new));
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
		dx = x - x_old;
		Eigen::VectorXd dg = g - g_old;
		double curv = dx.dot(dg);
		if (curv >= 1e-10 * dg.squaredNorm())
		{
			if (y_history.cols() == max_history)
			{
				matrix_ops::remove_end_column(y_history);
				matrix_ops::remove_end_column(s_history);
			}
			matrix_ops::add_end_column(y_history, dg);
			matrix_ops::add_end_column(s_history, dx);
			theta = dg.dot(dg) / dg.dot(dx);
		}
		if (debug && curv < 1e-10 * dg.squaredNorm())
		{
			std::cout << "optimise - negative curvature detected. Hessian update skipped\n";
		}
		alpha_init = std::min(1.0, g_old.norm() / g.norm());
	}
	return false;
}
