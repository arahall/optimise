#include <cmath>
#include <limits>
#include <stdexcept>

#include "brent.h"

double BRENT::root(function<double(double)> func, double a, double b, int max_iter, double atol, double ftol)
{
	constexpr double EPS = std::numeric_limits<double>::epsilon();

	double fa = func(a), fb = func(b);
	if (fa * fb > 0)
	{
		throw std::runtime_error("a and b must bracket a root\n");
	}

	if (fabs(fa) < fabs(fb))
	{
		swap(a, b);  swap(fa, fb);
	}
	double c = a, fc = fa, d = b, x;
	bool bisection = true;
	
	for (int iter = 0; iter < max_iter; ++iter)
	{
		if (fabs(a - b) <= atol)
		{
			return 0.5 * (a + b);
		}
		if (fabs(fa - fc) > ftol && fabs(fb - fc) > ftol)
		{
			x = a * fb * fc / ((fa - fb) * (fa - fc)) +
				b * fa * fc / ((fb - fa) * (fb - fc)) +
				c * fa * fb / ((fc - fa) * (fc - fb));
		}
		else
		{
			x = b - fb * (b - a) / (fb - fa);
		}
		double delta = fabs(2 * EPS * fabs(b));
		double min1 = fabs(x - b), min2 = fabs(b - c), min3 = fabs(c - d);
		if ((x < ((3 * a + b) / 4 && x > b)) ||
			(bisection && min1 >= 0.5 * min2) ||
			(!bisection && min1 >= 0.5 * min3) ||
			(bisection && min2 < delta) ||
			(!bisection && min3 < delta))
		{
			x = 0.5 * (a + b);
			bisection = true;
		}
		else
		{
			bisection = false;
		}
		double fnew = func(x);
		if (fabs(fnew) <= ftol)
		{
			return x;
		}
		d = c;
		c = b;
		if (fa * fnew < 0.0)
		{
			b = x; fb = fnew;
		}
		else
		{
			a = x;  fa = fnew;
		}
		if (fabs(fa) < fabs(fb))
		{
			swap(a, b); swap(fa, fb);
		}
	}
	throw std::runtime_error("maximum number of iterations exceeded in brent\n");
}