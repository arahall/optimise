#include <cmath>
#include <limits>
#include <stdexcept>
#include "ridders.h"


static inline double sign(double x)
{
	return copysign(1., x);
}
static inline bool converged(const double& xl, const double& xh, const double& atol, const double& rtol)
{
	double d = fabs(xh - xl);
	return d <= atol ? true : xl != 0.0 ? d / fabs(xl) <= rtol : false;
}
double RIDDERS::root(function<double(double)> func, double xl, double xh, int max_iter, double atol, 
					 double rtol, double ftol)
{
	if (xl > xh)
	{
		swap(xl, xh);
	}
	double fl = func(xl), fh = func(xh);
	if (fl * fh > 0)
	{
		throw std::runtime_error("xl and xh must bracket a root\n");
	}
	double xzero = std::numeric_limits<double>::infinity();
	
	for (int iter = 0; iter < max_iter; ++iter)
	{
		double xm = 0.5 * (xl + xh);
		double fm = func(xm);
		if (fabs(fm) <= ftol)
		{
			return xm;
		}
		double denom = sqrt(fm * fm - fl * fh);
		if (denom == 0.0)
		{
			throw std::runtime_error("denom is zero in ridders root finder. cannot proceed\n");
		}
		double xnew = xm + (xm - xl) * sign(fl - fh) * fm / denom;
		if (converged(xzero, xnew, atol, rtol))
		{
			if (converged(xl, xm, atol, rtol) || converged(xm, xh, atol, rtol))
			{
				return xzero;
			}
		}
		xzero = xnew;
		double fnew = func(xzero);
		if (fabs(fnew) <= ftol)
		{
			return xzero;
		}
		if (copysignf(fm, fnew) != fm)
		{
			xl = xm; fl = fm; xh = xzero; fh = fnew;
		}
		else if (copysignf(fm, fnew) != fl)
		{
			xh = xzero; fh = fnew;
		}
		else if (copysignf(fm, fnew) != fh)
		{
			xl = xzero; fl = fnew;
		}
		if (converged(xl, xh, atol, rtol))
		{
			return xzero;
		}
	}
	throw std::runtime_error("ridders::root did not converge");
}