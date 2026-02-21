#include <Eigen/Dense>
#include "nm_simplex.h"

namespace  // anonymous namespace rather than static functions
{
	void initialise(Eigen::MatrixXd& simplex, const Eigen::VectorXd& point)
	{
		int n = point.size();
		for (int i = 0; i < n + 1; i++)
		{
			simplex.row(i) = point.transpose();
		}

		for (int i = 0; i < n; i++)
		{
			simplex(i + 1, i) += (std::abs(point(i)) > 1e-8) ? 0.05 * point(i) : 0.05;
		}
	}
	void evaluate(const Eigen::MatrixXd& simplex, std::function<double(const Eigen::VectorXd&)> func,
				  Eigen::VectorXd& fx)
	{
		int n = simplex.rows();
		for (int i = 0; i < n; ++i)
		{
			fx(i) = func(simplex.row(i).transpose());
		}
	}
	void extremes(const Eigen::VectorXd& fx, int& ihi, int& ilo, int& inhi)
	{
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
	}
	void bearings(const Eigen::MatrixXd& simplex, Eigen::VectorXd& mid_point, Eigen::VectorXd& sline, int ihi)
	{
		int n = mid_point.size();
		mid_point.setZero();

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
		for (int j = 0; j < n; ++j)
		{
			mid_point(j) /= n;
			sline[j] = simplex(ihi, j) - mid_point(j);
		}
	}
	bool update(std::function<double(const Eigen::VectorXd&)> func,
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
	void contract(Eigen::MatrixXd& simplex, std::function<double(const Eigen::VectorXd&)> func,
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
	bool check_tol(const double& fmax, const double& fmin, const double& tol)
	{
		const double ZEPS = 0.0000000001;

		double delta = std::abs(fmax - fmin);
		double accuracy = (std::abs(fmax) + std::abs(fmin)) * tol;
		return delta < accuracy + ZEPS;
	}
}
bool NelderMeadSimplex::optimise(Eigen::VectorXd& point, 
								 std::function<double(const Eigen::VectorXd&)> func, 
								 double tol, int max_iter)
{
	int n = point.size();
	int ihi = 0, ilo = 0, inhi = 0;
	Eigen::MatrixXd simplex(n + 1, n);
	Eigen::VectorXd fx = Eigen::VectorXd::Zero(n + 1);
	Eigen::VectorXd mid = Eigen::VectorXd::Zero(n);
	Eigen::VectorXd line = Eigen::VectorXd::Ones(n);


	initialise(simplex, point);
	evaluate(simplex, func, fx);

	for (int iter = 0; iter < max_iter; ++iter)
	{
		extremes(fx, ihi, ilo, inhi);
		bearings(simplex, mid, line, ihi);

		if (check_tol(fx(ihi), fx(ilo), tol))
		{
			point = simplex.row(ilo);
			return true;
		}

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