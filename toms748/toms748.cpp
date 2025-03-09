#include <cmath>
#include <limits>
#include <stdexcept>

#include "toms748.h"

static inline double sign(double x)
{
	return copysign(1., x);
}
static void bracket(function<double(double)> func, 
					double &a, double &b, double &c, double &fa, double &fb, double tol, int neps, 
					double &d, double &fd)
{
	// adjust c if b-a is very small  or if c is close to a or b
	if (b - a <= 2 * tol)
	{
		c = 0.5 * (a + b);
	}
	else if (c <= a + tol)
	{
		c = a + tol;
	}
	else if (c >= b - tol)
	{
		c = b - tol;
	}
	double fc = func(c);
	if (fc == 0.0)
	{
		a = c; fa = 0.0; d = 0.0; fd = 0.0;
		return;
	}
	if (sign(fa) * sign(fc) < 0.0)
	{
		d = b; fd = fb; b = c; fb = fc;
	}
	else
	{
		d = a; fd = fa; a = c; fa = fc;
	}
}
static double newton_quad(double a, double b, double d, double fa, double fb, double fd, int nsteps)
{
	// uses nsteps to approximate the zero  in (a,b)
	double a0 = fa, a1 = (fb - fa) / (b - a), a2 = ((fd - fb) / (d - b) - a1) / (d - a);
	// safeguard to avoid overflow
	if (a2 == 0.0)
	{
		return a - a0 / a1;
	}
	double c = sign(a2) * sign(fa) > 0.0 ? a : b;
	// newton steps
	for (int step = 0; step < nsteps; ++step)
	{
		double pc = a0 + (a1 + a2 * (c - b) * (c - a));
		double pdc = a1 + a2 * (2. * c - (a + b));
		if (pdc == 0.0)
		{
			return a - a0 / a1;
		}
		c -= pc / pdc;
	}
	return c;
}
static double cubic_zero(double a, double b, double d, double e, double fa,
						 double fb, double fd, double fe)
{
	/* USES CUBIC INVERSE INTERPOLATION OF F(X) AT A, B, D, AND E TO
	 GET AN APPROXIMATE ROOT OF F(X).THIS PROCEDURE IS A SLIGHT
	 MODIFICATION OF AITKEN - NEVILLE ALGORITHM FOR INTERPOLATION
	 DESCRIBED BY STOER AND BULIRSCH IN "INTRO. TO NUMERICAL ANALYSIS"
	 SPRINGER - VERLAG.NEW YORK(1980).
	 */
	double q11 = (d - e) * fd / (fe - fd);
	double q21 = (b - d) * fb / (fd - fb);
	double q31 = (a - b) * fa / (fb - fa);
	double d21 = (b - d) * fd / (fd - fb);
	double d31 = (a - b) * fb / (fb - fa);
	double q22 = (d21 - q11) * fb / (fe - fb);
	double q32 = (d31 - q21) * fa / (fd - fa);
	double d32 = (d31 - q21) * fd / (fd - fa);
	double q33 = (d32 - q22) * fa / (fe - fa);
	return a + q31 + q32 + q33;
}
double TOMS748::root(function<double(double)> func, double a, double b, int neps, int max_iter, double tol)
{
	// finds a solution of f(x) = 0 in the interval a, b
	// the first iteration is a secant step. starting with the second iteration eithr a quadratic interpolation
	// or inverse cubic interpolation is taken. the third step is a double sized secand step
	// if the diameter of the enclosing intervat is still larger than 0.5*(b0 - a0) then an additional
	// bisection step is taken.

	const double mu = 0.5;
	if (a > b)
	{
		swap(a, b);
	}
	double fa = func(a);
	if (fa == 0)
	{
		return a;
	}
	double fb = func(b);
	if (fb == 0)
	{
		return b;
	}
	if (sign(fa) * sign(fb) > 0)
	{
		throw std::runtime_error("a, b must bracket a root\n");
	}
	double e = std::numeric_limits<double>::infinity(), fe = e, c, d, fd; 
	for (int iter = 0; iter < max_iter; ++iter)
	{
		double a0 = a; double b0 = b;
		//tol = fabs(fb) <= fabs(fa) ? get_tolerance(b, neps) : get_tolerance(a, neps);
		if (b - a <= tol)
		{
			return a;
		}
		// for the 1st iteration the secant step is taken
		if (iter == 0)
		{
			c = a - fa / (fb - fa) * (b - a);
			bracket(func, a, b, c, fa, fb, tol, neps, d, fd);
			if (fa == 0.0 || b - a <= tol)
			{
				return a;
			}
			continue;
		}
		double prof = (fa - fb) * (fa - fd) * (fa - fe) * (fb - fd) * (fb - fe) * (fd - fe);
		if (iter == 1 || prof == 0.0)
		{
			c = newton_quad(a, b, d, fa, fb, fd, 2);
		}
		else
		{
			c = cubic_zero(a, b, d, e, fa, fb, fd, fe);
			if ((c - a) * (c - b) >= 0.0)
			{
				c = newton_quad(a, b, d, fa, fb, fd, 2);
			}
		}
		e = d;
		fe = fd;
		bracket(func, a, b, c, fa, fb, tol, neps, d, fd);
		if (fa == 0.0 || b - a <= tol)
		{
			return a;
		}
		prof = (fa - fb) * (fa - fd) * (fa - fe) * (fb - fd) * (fb - fe) * (fd - fe);
		if (prof == 0.0)
		{
			c = newton_quad(a, b, d, fa, fb, fd, 3);
		}
		else
		{
			c = cubic_zero(a, b, d, e, fa, fb, fd, fe);
			if ((c - a) * (c - b) >= 0.0)
			{
				c = newton_quad(a, b, d, fa, fb, fd, 3);
			}
		}
		bracket(func, a, b, c, fa, fb, tol, neps, d, fd);
		if (fa == 0.0 || b - a <= tol)
		{
			return a;
		}
		e = d;
		fe = fd;
		// take a double sized secant step
		double u, fu;
		if (fabs(fa) < fabs(fb))
		{
			u = a; fu = fa;
		}
		else
		{
			u = b; fu = fb;
		}
		c = u - 2.0 * fu / (fb - fa) * (b - a);
		if (fabs(c - u) > 0.5 * (b - a))
		{
			c = a + 0.5 * (b - a);
		}
		bracket(func, a, b, c, fa, fb, tol, neps, d, fd);
		if (fa == 0.0 || b - a <= tol)
		{
			return a;
		}
		if (b - a < mu * (b0 - a0))
		{
			continue;
		}

		e = d; fe = fd;
		double temp_c = a + 0.5 * (b - a);
		bracket(func, a, b, temp_c, fa, fb, tol, neps, d, fd);

		if (fa == 0.0 || b - a <= tol)
		{
			return a;
		}
	}
	throw std::runtime_error("no convergenced");
}