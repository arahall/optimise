#include <iostream>

#include <vector>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <omp.h>

#include "lbfgsb.h"

using namespace std;

#if defined(USING_EIGEN)
#define EIGEN_NO_DEBUG
//#define EIGEN_DONT_PARALLELIZE
#define EIGEN_VECTORIZE
#else 
#include "lu.h"
#endif

#include "matrix_ops.h"
using namespace matrix_ops;

static vector<size_t> argsort(const VectorXd& x)
{
	vector <size_t> idx(x.size());
	iota(idx.begin(), idx.end(), 0);
	sort(idx.begin(), idx.end(), [&x](size_t i, size_t j) { return x[i] < x[j]; });
	return idx;
}

static double get_optimality(const VectorXd& x, const VectorXd& g,
							 const VectorXd& l, const VectorXd& u)
{
	size_t n = x.size();
	double max_element = 0.0;

	for (int i = 0; i < n; ++i)
	{
		double projected_value = std::min(std::max(l[i], x[i] - g[i]), u[i]);
		double difference = projected_value - x[i];
		max_element = std::max(max_element, fabs(difference));
	}

	return max_element;  // Return the maximum absolute value
}

static double find_alpha(const VectorXd& l, const VectorXd& u,
						 const VectorXd& xc, const VectorXd& du,
						 const vector<size_t>& free_vars_idx)
{
	double alpha_star = 1.0;
	size_t n = free_vars_idx.size();
	for (size_t i = 0; i < n; ++i)
	{
		size_t idx = free_vars_idx[i];
		if (du[i] > 0)
		{
			alpha_star = min(alpha_star, (u[idx] - xc[idx]) / du[i]);
		}
		else
		{
			alpha_star = min(alpha_star, (l[idx] - xc[idx]) / du[i]);
		}
	}
	return alpha_star;
}
static void get_break_points(const VectorXd& x, const VectorXd& g,
												  const VectorXd& l, const VectorXd& u,
												  VectorXd &t, VectorXd &d)
{
	// returns the break point vector and the search direction
	size_t n = x.size();
	if (t.size() != n || d.size() != n)
	{
		throw runtime_error("get_break_points - t and d must be the same size as x");
	}
	constexpr double realmax = numeric_limits<double>::max();
	constexpr double eps = numeric_limits<double>::epsilon();

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
#if defined(USING_EIGEN)
static tuple<VectorXd, VectorXd> get_cauchy_point(const VectorXd& x, const VectorXd& g,
												  const VectorXd& lb, const VectorXd& ub,
												  double theta, const MatrixXd& w, const MatrixXd& m)
{
	size_t n = x.size();
	VectorXd tt(n), d(n);
	get_break_points(x, g, lb, ub, tt, d);
	//
	vector<size_t> indices = argsort(tt);
	VectorXd xc = x;

	VectorXd p = w.transpose() * d;
	VectorXd c = VectorXd::Zero(w.cols());
	double fp = -d.dot(d);
	double fpp = -theta * fp - p.dot(m * p);
	double fpp0 = -theta * fp;
	double dt_min = -fp / fpp;
	double t_old = 0.0;
	// examine the rest of the segments
	double epsilon_fpp0 = numeric_limits<double>::epsilon() * fpp0;
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
		c += p * dt;
		double gb = g[b];
		VectorXd wb = w.row(b).transpose();
		fp += dt * fpp + gb * gb + theta * gb * zb - gb * wb.dot(m * c);
		fpp -= theta * gb * gb + 2.0 * gb * wb.dot(m * p) + gb * gb * wb.dot(m * wb);
		fpp = max(epsilon_fpp0, fpp);
		p += wb * gb;
		d[b] = 0.0;
		dt_min = -fp / fpp;
		t_old = t;
	}
	// perform final updates
	dt_min = max(dt_min, 0.0);
	t_old += dt_min;
	for (size_t j = k; j < n; ++j)
	{
		size_t idx = indices[j];
		xc[idx] += t_old * d[idx];
	}

	//c += p * dt_min;
	add_scale(c, p, dt_min, c);
	return make_tuple(xc, c);
}

static bool subspace_minimisation(const VectorXd& x, const VectorXd& g, const VectorXd& l, const VectorXd& u, 
								  const VectorXd& xc, const VectorXd& c, double theta, MatrixXd& w, const MatrixXd& m, 
								  VectorXd &xbar)
{
	size_t n = x.size();
	vector<size_t> free_vars_index;

#pragma omp parallel for
	for (size_t i = 0; i < n; ++i)
	{
		if (xc[i] > l[i] && xc[i] < u[i])
		{
#pragma omp critical
			free_vars_index.push_back(i);
		}
	}
	size_t num_free_vars = free_vars_index.size();
	if (num_free_vars == 0)
	{
		xbar = xc;
		return false;
	}
	MatrixXd wz(num_free_vars, c.size());
	wz = w(free_vars_index, Eigen::all);  // Directly extracting rows using index list

	// compute the reduced gradient of mk restricted to free variables
	// rr = g + theta * (xc - x) - w *(m*c)
	VectorXd rr = g + (xc - x) * theta - w * (m * c);
	VectorXd r(num_free_vars);
	for (int i = 0; i < num_free_vars; ++i)
	{
		r[i] = rr[free_vars_index[i]];
	}
	// form intermediate variables
	
	double one_over_theta = 1.0 / theta;
	MatrixXd wz_T = wz.transpose();
	VectorXd v = m * wz_T * r;
	MatrixXd wz_T_wz = wz_T * wz;
	MatrixXd big_n = MatrixXd::Identity(wz_T_wz.rows(), wz_T_wz.cols());
	big_n.noalias() -= m * wz_T_wz * one_over_theta;  // Avoid aliasing
	v = big_n.colPivHouseholderQr().solve(v); 
	VectorXd du = -one_over_theta * r - one_over_theta * one_over_theta * wz * v;

	// find alpha star
	double alpha_star = find_alpha(l, u, xc, du, free_vars_index);

	// compute the subspace minimisation
	xbar = xc;
	for (size_t i = 0; i < num_free_vars; ++i)
	{
		size_t idx = free_vars_index[i];
		xbar[idx] += alpha_star * du[i];
	}
	return true;
}

static double alpha_zoom(function<double(const VectorXd&)> func,
						 function<VectorXd(const VectorXd&)> gradient,
						 const VectorXd& x0,
						 double f0, const VectorXd& g0, const VectorXd& p,
						 double alpha_lo, double alpha_hi,
						 int max_iters, double c1, double c2)
{
	size_t n = x0.size();
	VectorXd x(n), x_lo(n);
	
	double dphi0 = g0.dot(p);

	for (int i = 0; i < max_iters; ++i)
	{
		double alpha_i = 0.5 * (alpha_lo + alpha_hi);
		add_scale(x0, p, alpha_i, x);
		double f_i = func(x);
		add_scale(x0, p, alpha_lo, x_lo);
		double f_lo = func(x_lo);

		if ((f_i > f0 + c1 * alpha_i * dphi0) || (f_i >= f_lo))
		{
			alpha_hi = alpha_i;
		}
		else
		{
			double dphi = gradient(x).dot(p);
			if (fabs(dphi) <= -c2 * dphi0)
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
static double strong_wolfe(function<double(const VectorXd&)> func,
						   function<VectorXd(const VectorXd&)> gradient,
						   const VectorXd& x0, double f0, const VectorXd& g0,
						   const VectorXd& p, int max_iters, double c1, double c2, double alpha_max)
{
	// compute line search satisfying strong Wolfe conditions
	double f_im1 = f0, alpha_im1 = 0.0, alpha_i = 1.0;
	double dphi0 = g0.dot(p);
	int n = x0.size();
	VectorXd x(n);

	for (int iter = 0; iter < max_iters; ++iter)
	{
		add_scale(x0, p, alpha_i, x);
		double f_i = func(x);
		if ((f_i > f0 + c1 * dphi0) || (iter > 1 && f_i >= f_im1))
		{
			return alpha_zoom(func, gradient, x0, f0, g0, p, alpha_im1, alpha_i, max_iters, c1, c2);
		}
		VectorXd g_i = gradient(x);
		double dphi = g_i.dot(p);
		if (fabs(dphi) <= -c2 * dphi0)
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
static MatrixXd hessian(const MatrixXd& s, const MatrixXd& y, double theta)
{
	MatrixXd sT = s.transpose();

	// Compute the matrix a and its lower triangular and diagonal parts
	MatrixXd a = sT * y;
	MatrixXd l = a.triangularView<Eigen::StrictlyLower>();
	MatrixXd d = -1 * a.diagonal().asDiagonal();

	// Preallocate the final matrix mm
	MatrixXd mm(d.rows() + l.rows(), d.cols() + l.rows());

	// Fill the top part of mm (d and l.transpose())
	mm.topLeftCorner(d.rows(), d.cols()) = d;
	mm.topRightCorner(d.rows(), l.rows()) = l.transpose();

	// Fill the bottom part of mm (l and theta * s.transpose() * s)
	mm.bottomLeftCorner(l.rows(), l.cols()) = l;
	mm.bottomRightCorner(l.rows(), s.cols()).noalias() = theta * sT * s;

	// Compute the inverse of the full matrix mm
	return mm.partialPivLu().inverse();
}
bool LBFGSB::optimize(function<double(const VectorXd&)> func,
					  function<VectorXd(const VectorXd&)> gradient,
					  VectorXd& x, const VectorXd& lb, const VectorXd& ub,
					  int max_history, int max_iter, int ln_srch_maxiter, 
					  double tol, double c1, double c2, double alpha_max,
					  double eps_factor, bool debug)
{
	// func - the function to be minimised
	// gradient - the gradient of the function to be minimosed
	// x - the solution 
	// max_history - number of corrections used in the limited memeory matrix
	// max_history < 3 not recommended, large m not recommend
	// 3 <= m < 20 is the recommended range for me
	//
	if (debug)
	{
		std::cout << "max_history: " << max_history << " max_iter: " << max_iter << " ln_srch_maxiter: " << ln_srch_maxiter <<
			" tol: " << tol << " c1: " << c1 << " c2: " << c2 << " alpha_max: " << alpha_max << endl;
	}
	size_t n = x.size(); // the problem dimension
	double tol_f = std::numeric_limits<double>::epsilon() * eps_factor;
	// check that the bounds are well specified
	for (unsigned i = 0; i < n; ++i)
	{
		if (lb[i] >= ub[i])
		{
			throw runtime_error("LBFGSB::optimise - lower bound must be less than upper boound");
		}
	}
	
	MatrixXd w = MatrixXd::Zero(n, 1), m = MatrixXd::Zero(1, 1);
	MatrixXd y_history, s_history;
	VectorXd xbar(n);
	
	constexpr double eps = numeric_limits<double>::epsilon();
	double f = func(x);
	VectorXd g = gradient(x);
	if (g.size() != n)
	{
		throw runtime_error("LBFGSB::optimise - length of gradient must be the same as the problem dimension");
	}
	double theta = 1.0;
	for (int iter = 0; iter < max_iter; ++iter)
	{
		double opt = get_optimality(x, g, lb, ub);
		if (debug)
		{
			cout << "optimality = " << opt << " func = " << f << "\n";
			for (const auto& x_ : x)
			{
				cout << x_ << " ";
			}
			cout << "\n";
		}
		if (opt < tol)
		{
			if (debug)
			{
				cout << "converged in " << iter << " iterations\n";
			}
			return true;
		}
		VectorXd x_old(x);
		VectorXd g_old(g);

		// compute new search directon
		tuple<VectorXd, VectorXd> cp = get_cauchy_point(x, g, lb, ub, theta, w, m);
		VectorXd xc = get<0>(cp);
		VectorXd c = get<1>(cp);
		bool flag = subspace_minimisation(x, g, lb, ub, xc, c, theta, w, m, xbar);
		VectorXd dx = xbar - x;
		double alpha = flag ? strong_wolfe(func, gradient, x, f, g, dx, ln_srch_maxiter, c1, c2, alpha_max) : 1.0;
		x += alpha * dx;
		double f_new = func(x);
		if (debug)
		{
			std::cout << "f_new: " << f_new << " f: " << f << "\n";
		}
		double f_tol_check = fabs(f_new - f) / max(max(abs(f), 1.0), abs(f_new));
		if (f_tol_check <= tol_f)
		{
			if (debug)
			{
				cout << "converged in " << iter << "iterations due to function tolerance: " <<
					f_tol_check << "tolf: " << tol_f << "\n";
				return true;
			}
		}
		f = f_new;
		g = gradient(x);
		dx = x - x_old;
		VectorXd dg = g - g_old;
		double curv = dx.dot(dg);
		if (curv >= eps)
		{
			if (y_history.cols() == max_history)
			{
				remove_end_column(y_history);
				remove_end_column(s_history);
			}
			add_end_column(y_history, dg);
			add_end_column(s_history, dx);
			theta = dg.dot(dg) / dg.dot(dx);
			w = MatrixXd(y_history.rows(), y_history.cols() + s_history.cols());
			w << y_history, theta* s_history;
			m = hessian(s_history, y_history, theta);
		}
		if (debug && curv < numeric_limits<double>::epsilon())
		{
			cout << "optimise - negative curvature detected. Hessian update skipped\n";
		}
	}
	return false;
}
#else
static tuple<VectorXd, VectorXd> get_cauchy_point(const VectorXd& x, const VectorXd& g,
												  const VectorXd& lb, const VectorXd& ub,
												  double theta, vector<VectorXd>& w,
												  const vector<VectorXd>& m)
{
	// w(n, 2l)
	// m(2l, 2l)
	size_t n = x.size();
	VectorXd tt(n), d(n);
	get_break_points(x, g, lb, ub, tt, d);
	//
	vector<size_t> indices = argsort(tt);
	VectorXd xc(x);

	VectorXd p = multiply(transpose(w), d);
	size_t cols = w[0].size();
	VectorXd c(cols, 0.0);
	double fp = -dot(d, d);
	double fpp = -theta * fp - dot(p, multiply(m, p));
	double fpp0 = -theta * fp;
	double dt_min = -fp / fpp;
	double t_old = 0.0;
	// examine the rest of the segments
	int k = 0;
	for (int i = 0; i < n; ++i)
	{
		size_t b = indices[i];
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
		c = add(c, scale(p, dt));
		double gb = g[b];
		VectorXd wb = get_row(w, b);
		fp += dt * fpp + gb * gb + theta * gb * zb - gb * dot(wb, multiply(m, c));
		fpp -= theta * gb * gb + 2.0 * gb * dot(wb, multiply(m, p)) + gb * gb * dot(wb, multiply(m, wb));
		fpp = max(numeric_limits<double>::epsilon() * fpp0, fpp);
		p = add(p, scale(wb, gb));
		d[b] = 0.0;
		dt_min = -fp / fpp;
		t_old = t;
	}
	// perform final updates
	dt_min = max(dt_min, 0.0);
	t_old += dt_min;
	for (size_t j = k; j < xc.size(); ++j)
	{
		size_t idx = indices[j];
		xc[idx] += t_old * d[idx];
	}
	add_scale(c, p, dt_min, c);
	//c = add(c, scale(p, dt_min));
	return make_tuple(xc, c);
}
static bool subspace_minimisation(const VectorXd& x, const VectorXd& g,
												   const VectorXd& l, const VectorXd& u,
												   const VectorXd& xc, const VectorXd& c,
												   double theta, const vector<VectorXd>& w,
												   const vector<VectorXd>& m, VectorXd &xbar)
{
	size_t n = x.size();
	vector<size_t> free_vars_index;
#pragma omp parallel for
	for (size_t i = 0; i < n; ++i)
	{
		if (xc[i] > l[i] && xc[i] < u[i])
		{
#pragma omp critical
			free_vars_index.push_back(i);
		}
	}
	size_t num_free_vars = free_vars_index.size();
	if (num_free_vars == 0)
	{
		xbar = xc;
		return false;
	}
	vector<VectorXd> wz(num_free_vars);
	for (size_t i = 0; i < num_free_vars; ++i)
	{
		wz[i].resize(c.size());
		size_t idx = free_vars_index[i];
		for (size_t j = 0; j < c.size(); ++j)
		{
			wz[i][j] = w[idx][j];
		}
	}
	vector<VectorXd> wtz = transpose(wz);
	// compute the reduced gradient of mk restricted to free variables
	// rr = g + theta * (xc - x) - w *(m*c)
	VectorXd temp1 = scale(subtract(xc, x), theta);
	VectorXd temp2 = multiply(w, multiply(m, c));
	VectorXd rr = add(g, subtract(temp1, temp2));
	VectorXd r(num_free_vars);
	for (int i = 0; i < num_free_vars; ++i)
	{
		r[i] = rr[free_vars_index[i]];
	}
	// form intermediate variables
	double one_over_theta = 1.0 / theta;
	VectorXd v = multiply(m, multiply(wtz, r));
	vector<VectorXd> big_n = scale(multiply(wtz, wz), one_over_theta);
	big_n = subtract(identity(big_n.size()), multiply(m, big_n));
	vector<size_t> p(big_n.size() + 1);
	lu::decompose(big_n, p);
	v = lu::solve(big_n, p, v);
	VectorXd du = add(scale(r, -one_over_theta), scale(multiply(wz, v), -one_over_theta * one_over_theta));

	// find alpha star
	double alpha_star = find_alpha(l, u, xc, du, free_vars_index);

	// compute the subspace minimisation
	xbar = xc;
	for (size_t i = 0; i < num_free_vars; ++i)
	{
		size_t idx = free_vars_index[i];
		xbar[idx] += alpha_star * du[i];
	}
}
static double alpha_zoom(function<double(const VectorXd&)> func,
						 function<VectorXd(const VectorXd&)> gradient,
						 const VectorXd& x0,
						 double f0, const VectorXd& g0, const VectorXd& p,
						 double alpha_lo, double alpha_hi,
						 int max_iters, double c1, double c2)
{
	int n = x0.size();
	double dphi0 = dot(g0, p);
	double alpha_i = 0.5 * (alpha_lo + alpha_hi);
	VectorXd x(n), x_lo(n);

	for (int i = 0; i < max_iters; ++i)
	{
		add_scale(x0, p, alpha_i, x);
		double f_i = func(x);
		add_scale(x0, p, alpha_lo, x_lo);
		double f_lo = func(x_lo);

		if ((f_i > f0 + c1 * alpha_i * dphi0) || (f_i >= f_lo))
		{
			alpha_hi = alpha_i;
		}
		else
		{
			double dphi = dot(gradient(x), p);
			if (fabs(dphi) <= -c2 * dphi0)
			{
				return alpha_i;
			}
			if (dphi * (alpha_hi - alpha_lo) >= 0)
			{
				alpha_hi = alpha_lo;
			}
			alpha_lo = alpha_i;
		}
		alpha_i = 0.5 * (alpha_hi + alpha_lo);
	}
	return alpha_i;
}
static double strong_wolfe(function<double(const VectorXd&)> func,
						   function<VectorXd(const VectorXd&)> gradient,
						   const VectorXd& x0, double f0, const VectorXd& g0,
						   const VectorXd& p, int max_iters, double c1, double c2, double alpha_max)
{
	// compute line search satisfying strong Wolfe conditions
	double f_im1 = f0, alpha_im1 = 0.0, alpha_i = 1.0;
	double dphi0 = dot(g0, p);
	int n = x0.size();
	VectorXd x(n);

	for (int iter = 0; iter < max_iters; ++iter)
	{
		add_scale(x0, p, alpha_i, x);
		double f_i = func(x);
		if ((f_i > f0 + c1 * dphi0) || (iter > 1 && f_i >= f_im1))
		{
			return alpha_zoom(func, gradient, x0, f0, g0, p, alpha_im1, alpha_i, max_iters, c1, c2);
		}
		VectorXd g_i = gradient(x);
		double dphi = dot(g_i, p);
		if (fabs(dphi) <= -c2 * dphi0)
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
static vector<VectorXd> hessian(const vector<VectorXd>& s, const vector<VectorXd>& y, double theta)
{
	vector<VectorXd> st = transpose(s);
	vector<VectorXd> a = multiply(st, y);
	vector<VectorXd> l = tril(a, -1);
	vector<VectorXd> d = scale(diag(diag(a)), -1);
	// form the upper part
	vector<VectorXd> lt = transpose(l);
	vector<VectorXd> sts = multiply(st, s);
	vector<VectorXd> top = hconcat(d, lt);
	vector<VectorXd> bottom = hconcat(l, scale(sts, theta));
	vector<VectorXd> mm = vconcat(top, bottom);
	try
	{
		vector<VectorXd> m = lu::inverse(mm);
		return m;
	}
	catch (std::runtime_error& e)
	{
		throw e;
	}
}
bool LBFGSB::optimize(function<double(const VectorXd&)> func,
					  function<VectorXd(const VectorXd&)> gradient,
					  VectorXd& x, const VectorXd& lb, const VectorXd& ub,
					  int max_history, int max_iter, int ln_srch_maxiter, 
					  double tol, double c1, double c2, double alpha_max, double eps_factor, bool debug)
{
	// func - the function to be minimised
	// gradient - the gradient of the function to be minimosed
	// x - the solution 
	// max_history - number of corrections used in the limited memeory matrix
	// max_history < 3 not recommended, large m not recommend
	// 3 <= m < 20 is the recommended range for me
	//
	size_t n = x.size(); // the problem dimension
	double tol_f = std::numeric_limits<double>::epsilon() * eps_factor;

	VectorXd xbar(n);
	vector<VectorXd> y_history, s_history;
	vector<VectorXd> w(n, VectorXd(1, 0.0)), m(1, VectorXd(1, 0.0));
	if (debug)
	{
		cout << "running non-Eigen version\n";
	}
	// check that the bounds are well specified
	for (unsigned i = 0; i < n; ++i)
	{
		if (lb[i] >= ub[i])
		{
			throw runtime_error("LBFGSB::optimise - lower bound must be less than upper boound");
		}
	}
	double f = func(x);
	VectorXd g = gradient(x);
	if (g.size() != n)
	{
		throw runtime_error("LBFGSB::optimise - len of gradient must be the same as the problem dimension");
	}
	double theta = 1.0;
	for (int iter = 0; iter < max_iter; ++iter)
	{
		double opt = get_optimality(x, g, lb, ub);
		if (debug)
		{
			cout << "optimality = " << opt << " func = " << f << "\n";
			for (int i = 0; i < min((int)n,10); ++i)
			{
				cout << x[i] << " ";
			}
			cout << "\n";
		}
		if (opt < tol)
		{
			if (debug)
			{
				cout << "converged in " << iter << " iterations\n";
			}
			return true;
		}
		VectorXd x_old(x);
		VectorXd g_old(g);

		// compute new search directon
		tuple<VectorXd, VectorXd> cp = get_cauchy_point(x, g, lb, ub, theta, w, m);
		VectorXd xc = get<0>(cp);
		VectorXd c = get<1>(cp);
		bool flag = subspace_minimisation(x, g, lb, ub, xc, c, theta, w, m, xbar);
		double alpha = flag ? strong_wolfe(func, gradient, x, f, g, subtract(xbar, x), ln_srch_maxiter, c1, c2, alpha_max) : 1.0;
		x = add(x, scale(subtract(xbar, x), alpha));
		double f_new = func(x);
		if (debug)
		{
			std::cout << "f_new: " << f_new << " f: " << f << "\n";
		}
		double f_tol_check = fabs(f_new - f) / max(max(abs(f), 1.0), abs(f_new));
		if (f_tol_check <= tol_f)
		{
			if (debug)
			{
				cout << "converged in " << iter << "iterations due to function tolerance: " << 
					f_tol_check << "tolf: " << tol_f << "\n";
				return true;
			}
		}
		f = f_new;
		g = gradient(x);
		VectorXd dx = subtract(x, x_old);
		VectorXd dg = subtract(g, g_old);
		double curv = dot(dx, dg);
		if (curv >= numeric_limits<double>::epsilon())
		{
			if (num_columns(y_history) == max_history)
			{
				matrix_ops::remove_end_column(y_history);
				matrix_ops::remove_end_column(s_history);
			}
			add_end_column(y_history, dg);
			add_end_column(s_history, dx);
			theta = dot(dg, dg) / dot(dg, dx);
			w = hconcat(y_history, scale(s_history, theta));
			try
			{
				m = hessian(s_history, y_history, theta);
			}
			catch (...)
			{
				if (debug)
				{
					cout << "hessian update failed\n";
				}
				return false;
			}
		}
		else
		{
			if (debug)
			{
				cout << "negative curvature detected - skipping BFGS update\n";
			}
		}
	}
	return false;
}
#endif