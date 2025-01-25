#ifndef __GRADIENT_STRATEGY_H
#define __GRADIENT_STRATEGY_H

#include <functional>

using namespace std;
#include "matrix_ops.h"

namespace gradients
{
    class GradientStrategy
    {
    public:
        // Pure virtual function using the () operator for gradient calculation
        virtual VectorXd operator()(const VectorXd& x) const = 0;

        // Virtual destructor to ensure proper cleanup of derived classes
        virtual ~GradientStrategy() = default;
    };

    class NumericalGradient : public GradientStrategy
    {
    private:
        function<double(const VectorXd&)> func;  // Objective function
        double epsilon;
    public:
        NumericalGradient(function<double(const VectorXd&)> func, double epsilon = 1e-4)
            : func(func), epsilon(epsilon) {}

        // Overload the () operator
        VectorXd operator()(const VectorXd& x) const override
        {
            VectorXd grad(x.size());
            VectorXd x_plus = x;

            for (size_t i = 0; i < x.size(); ++i)
            {
                double original_value = x[i];
                x_plus[i] = original_value + epsilon;
                double f_plus = func(x_plus);

                x_plus[i] = original_value - epsilon;
                double f_minus = func(x_plus);

                grad[i] = (f_plus - f_minus) / (2 * epsilon);
                x_plus[i] = original_value;  // Restore original value
            }

            return grad;
        }
    };
    class AnalyticalGradient : public GradientStrategy
    {
    private:
        function<VectorXd(const VectorXd&)> grad_func;

    public:
        AnalyticalGradient(function<VectorXd(const VectorXd&)> grad_func)
            : grad_func(grad_func) {}

        // Overload the () operator
        VectorXd operator()(const VectorXd& x) const override
        {
            return grad_func(x);  // Return the gradient using the provided function
        }
    };
};
#endif
