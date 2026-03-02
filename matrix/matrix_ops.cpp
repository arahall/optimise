#include <algorithm>
#include <numeric>

#include <Eigen/Dense>

#include "matrix_ops.h"

void matrix_ops::remove_end_column(Eigen::MatrixXd& a)
{
    a.block(0, 0, a.rows(), a.cols() - 1) = a.leftCols(a.cols() - 1);
    a.conservativeResize(a.rows(), a.cols() - 1);
}
void matrix_ops::add_end_column(Eigen::MatrixXd& a, const Eigen::VectorXd& v)
{
    if (a.size() == 0)
    {
        a.resize(v.size(), 1);
    }
    else
    {
        a.conservativeResize(a.rows(), a.cols() + 1);
    }
    a.col(a.cols() - 1) = v;
}
void matrix_ops::add_scale(const Eigen::VectorXd& a, const Eigen::VectorXd& x, const double& b, Eigen::VectorXd& y)
{
    std::transform(a.begin(), a.end(), x.begin(), y.begin(),
              [b](double ai, double xi) {
                  return ai + b * xi;
              });
}
std::vector<size_t> matrix_ops::argsort(const Eigen::VectorXd& x)
{
    std::vector <size_t> idx(x.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&x](size_t i, size_t j) { return x[i] < x[j]; });
    return idx;
}