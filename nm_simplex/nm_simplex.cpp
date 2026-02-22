#include <Eigen/Dense>
#include "nm_simplex.h"

namespace  // anonymous namespace rather than static functions
{
	Eigen::MatrixXd initialise(const Eigen::VectorXd& point)
	{
		int n = point.size();
		Eigen::MatrixXd simplex(n + 1, n);
		for (int i = 0; i < n + 1; i++)
		{
			simplex.row(i) = point.transpose();
		}

		for (int i = 0; i < n; i++)
		{
			simplex(i + 1, i) += (std::abs(point(i)) > 1e-8) ? 0.05 * point(i) : 0.05;
		}
		return simplex;
	}
	Eigen::VectorXd evaluate(const Eigen::MatrixXd& simplex, const std::function<double(const Eigen::VectorXd&)> &func)
	{
		int n = simplex.rows();
		Eigen::VectorXd fx(n);
		for (int i = 0; i < n; ++i)
		{
			fx(i) = func(simplex.row(i).transpose());
		}
		return fx;
	}
	std::tuple<int, int, int> extremes(const Eigen::VectorXd& fx)
	{
		int ilo, ihi, inhi;
		if (fx(0) > fx(1))
		{
			ihi = 0; ilo = 1; inhi = 0;
		}
		else
		{
			ihi = 1; ilo = 0; inhi = 1;
		}
		for (int i = 2; i < fx.size(); ++i)
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
	std::tuple<Eigen::VectorXd, Eigen::VectorXd> bearings(const Eigen::MatrixXd& simplex, int ihi)
	{
		int n = simplex.rows() - 1;
		Eigen::VectorXd mid_point = Eigen::VectorXd::Zero(n);

		for (int i = 0; i < n + 1; ++i)
		{
			if (i != ihi)
			{
				for (int j = 0; j < n; ++j)
				{
					mid_point(j) += simplex(i, j);
				}
			}
		}
		Eigen::VectorXd sline(n);
		for (int j = 0; j < n; ++j)
		{
			mid_point(j) /= n;
			sline(j) = simplex(ihi, j) - mid_point(j);
		}
		return { mid_point, sline };
	}
	bool update(const std::function<double(const Eigen::VectorXd&)> &func,
				Eigen::MatrixXd& simplex, Eigen::VectorXd& fx,
				const Eigen::VectorXd& mid_point,
				const Eigen::VectorXd& sline,
				int ihi,
				double scale_factor)
	{
		Eigen::VectorXd next_val = mid_point + scale_factor * sline;
		double f = func(next_val);
		if (f >= fx(ihi))
		{
			return false;
		}

		simplex.row(ihi) = next_val.transpose();
		fx(ihi) = f;

		return true;
	}
	void contract(Eigen::MatrixXd& simplex, const std::function<double(const Eigen::VectorXd&)> &func,
				  Eigen::VectorXd& fx, int ilo)
	{
		int n = simplex.rows();
		for (int i = 0; i < n; ++i)
		{
			if (i != ilo)
			{
				simplex.row(i) = 0.5 * (simplex.row(i) + simplex.row(ilo));
				fx(i) = func(simplex.row(i).transpose());
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
}
bool NelderMeadSimplex::optimise(std::function<double(const Eigen::VectorXd&)> func, 
								 Eigen::VectorXd& point,
								 double tol, int max_iter)
{
	if (point.size() == 0)
	{
		throw std::runtime_error("Nelder Mead Simplex - empty initial vector supplied\n");
	}
	int ilo = 0;
	Eigen::MatrixXd simplex = initialise(point);
	Eigen::VectorXd fx = evaluate(simplex, func);

	for (int iter = 0; iter < max_iter; ++iter)
	{
		int ihi, inhi;
		std::tie(ilo, ihi, inhi) = extremes(fx);
		if (check_tol(fx(ihi), fx(ilo), tol))
		{
			point = simplex.row(ilo);
			return true;
		}
		Eigen::VectorXd mid, line;
		std::tie(mid, line) = bearings(simplex, ihi);
		bool reflected = update(func, simplex, fx, mid, line, ihi, -1.0);

		if (reflected && fx(ihi) < fx(ilo))
		{
			update(func, simplex, fx, mid, line, ihi, -2.0);
		}
		else if (fx(ihi) >= fx(inhi))
		{
			if (!update(func, simplex, fx, mid, line, ihi, 0.5))
			{
				contract(simplex, func, fx, ilo);
			}
		}
	}
	point = simplex.row(ilo).transpose();
	return false;
}