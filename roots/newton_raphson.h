#ifndef __NEWTON_RAPHSON_H
#define __NEWTON_RAPHSON_H

#include <stdexcept>
#include <cmath>

namespace NEWTON_RAPHSON
{
    template <typename Func>
    double root(Func func, double guess, unsigned max_iter, double ftol, double xtol)
    {
        for (unsigned i = 0; i < max_iter; ++i)
        {
            auto f = func(guess);
            double step = f.first / f.second;
            guess -= step;
            if (abs(step) < xtol || abs(f.first) < ftol)
            {
                return guess;
            }
        }
        throw std::runtime_error("Newton-Raphson method did not converge");
    }
}
#endif
