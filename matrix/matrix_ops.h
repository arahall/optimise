#ifndef __MATRIX_OPS
#define __MATRIX_OPS

#include <vector>
#include <Eigen/Dense>

namespace matrix_ops
{
    void add_end_column(Eigen::MatrixXd& a, const Eigen::VectorXd& v);
    void remove_end_column(Eigen::MatrixXd& a);
    void add_scale(const Eigen::VectorXd& a, const Eigen::VectorXd& x, const double& b, Eigen::VectorXd& y);
    std::vector<size_t> argsort(const Eigen::VectorXd& x);
};
#endif