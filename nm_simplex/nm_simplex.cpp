#include <Eigen/Dense>
#include "nm_simplex.h"

namespace  // anonymous namespace rather than static functions
{
	Eigen::MatrixXd initialise(const Eigen::VectorXd& point)
	{
		int n = point.size();
		Eigen::MatrixXd simplex(n, n+1);
		simplex.colwise() = point;
		for (Eigen::Index i = 0; i < n; ++i)
		{
			simplex(i, i+1) += (std::abs(point(i)) > 1e-8) ? 0.05 * point(i) : 0.05;
		}
		return simplex;
	}
	Eigen::VectorXd evaluate(const Eigen::MatrixXd& simplex, const std::function<double(const Eigen::VectorXd&)>& func)
	{
		int n = simplex.cols();
		Eigen::VectorXd fx(n);
		for (Eigen::Index i = 0; i < n; ++i)
		{
			fx(i) = func(simplex.col(i));
		}
		return fx;
	}
	std::tuple<Eigen::Index, Eigen::Index, Eigen::Index> extremes(const Eigen::VectorXd& fx)
	{
		Eigen::Index ihi = (fx(0) > fx(1)) ? 0 : 1;
		Eigen::Index ilo = ihi == 0 ? 1 : 0;
		Eigen::Index inhi = ilo;  // the lesser of the first two — worst case for inhi

		for (Eigen::Index i = 2; i < fx.size(); ++i) 
		{
			if (fx(i) <= fx(ilo))
			{
				ilo = i;
			}
			else if (fx(i) > fx(ihi))
			{
				inhi = ihi; ihi = i;
			}
			else if (fx(i) > fx(inhi))
			{
				inhi = i;
			}
		}
		return { ilo, ihi, inhi };
	}
	Eigen::VectorXd centroid(const Eigen::MatrixXd& simplex, Eigen::Index ihi)
	{
		int n = simplex.rows();
		Eigen::VectorXd c = Eigen::VectorXd::Zero(n);
		for (Eigen::Index i = 0; i < n + 1; ++i)
		{
			if (i != ihi)
			{
				c += simplex.col(i);
			}
		}
		c /= n;
		return c;
	}
	void contract(Eigen::MatrixXd& simplex, Eigen::Index ilo, double sigma)
	{
		int n = simplex.cols();
		Eigen::VectorXd x1 = simplex.col(ilo);
		for (Eigen::Index i = 0; i < n; ++i)
		{
			if (i != ilo)
			{
				simplex.col(i) = x1 + sigma * (simplex.col(i) - x1);
			}
		}
	}
	bool check_tol(double fmax, double fmin, double tol)
	{
		const double ZEPS = 0.0000000001;

		double delta = std::abs(fmax - fmin);
		double accuracy = (std::abs(fmax) + std::abs(fmin)) * tol;
		return delta < accuracy + ZEPS;
	}

	bool amoeba(const std::function<double(const Eigen::VectorXd&)>& func,
				 Eigen::VectorXd& point, double alpha, double gamma, double rho, double sigma,
				 double tol, int max_iter)
	{
		Eigen::Index ilo, ihi, inhi;
		Eigen::MatrixXd simplex = initialise(point);
		for (int iter = 0; iter < max_iter; ++iter)
		{
			Eigen::VectorXd fx = evaluate(simplex, func);
			std::tie(ilo, ihi, inhi) = extremes(fx);

			if (check_tol(fx(ihi), fx(ilo), tol))
			{
				point = simplex.col(ilo);
				return true;
			}

			Eigen::VectorXd x0 = centroid(simplex, ihi);
			Eigen::VectorXd xr = x0 + alpha * (x0 - simplex.col(ihi));
			double fx_r = func(xr);
			if (fx_r >= fx(ilo) && fx_r < fx(inhi))
			{
				// reflection
				simplex.col(ihi) = xr;
			}
			else if (fx_r < fx(ilo))
			{
				// expansion
				Eigen::VectorXd xe = x0 + gamma * (xr - x0);
				double fx_e = func(xe);
				if (fx_e < fx_r)
				{
					simplex.col(ihi) = xe;
				}
				else
				{
					simplex.col(ihi) = xr;
				}
			}
			else
			{
				// contraction
				bool use_reflection = fx_r < fx(ihi);
				Eigen::VectorXd x_contract = use_reflection ? xr : simplex.col(ihi);
				double fx_contract_limit = use_reflection ? fx_r : fx(ihi);
				Eigen::VectorXd xc = x0 + rho * (x_contract - x0);
				double fx_c = func(xc);
				if (fx_c < fx_contract_limit)
				{
					simplex.col(ihi) = xc;
				}
				else
				{
					contract(simplex, ilo, sigma);
				}
			}
		}
		Eigen::VectorXd fx = evaluate(simplex, func);
		std::tie(ilo, ihi, inhi) = extremes(fx);
		point = simplex.col(ilo);
		return false;
	}
}
bool NelderMeadSimplex::optimize(const std::function<double(const Eigen::VectorXd&)>& func,
								 Eigen::VectorXd& point, double alpha, double gamma, double rho, double sigma,
								 double tol, int max_iter, int max_restarts)
{
	if (point.size() == 0)
	{
		throw std::runtime_error("Nelder Mead Simplex - empty initial vector supplied\n");
	}
	for (int restart = 0; restart < max_restarts; ++restart)
	{
		if (amoeba(func, point, alpha, gamma, rho, sigma, tol, max_iter))
		{
			return true;
		}
	}
	return false;
}