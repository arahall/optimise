#ifndef __MATRIX_OPS
#define __MATRIX_OPS

using namespace std;
#include <vector>
#if defined(USING_EIGEN)
#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::VectorXd;
#else
typedef vector<double> VectorXd;
typedef vector<vector<double>> MatrixXd;
#endif
namespace matrix_ops
{
#if !defined(USING_EIGEN)
    MatrixXd identity(size_t n);
    MatrixXd tril(const MatrixXd& a, int k = 0);
    VectorXd diag(const MatrixXd& a);
    MatrixXd diag(const VectorXd& d);
    void initialise(MatrixXd& a, const VectorXd& v);
    size_t num_columns(const MatrixXd& a);
    size_t num_rows(const MatrixXd& a);
    VectorXd scale(const VectorXd& x, double a);
    MatrixXd scale(const MatrixXd& x, double a);
    MatrixXd copy(const MatrixXd& a_);
    MatrixXd vconcat(const MatrixXd& a, const MatrixXd& b);
    MatrixXd hconcat(const MatrixXd& a, const MatrixXd& b);
    VectorXd get_col(const MatrixXd& a, size_t col);
    VectorXd get_row(const MatrixXd& a, size_t row);
    MatrixXd transpose(const MatrixXd& a);
    VectorXd multiply(const MatrixXd& a, const VectorXd& x);
    MatrixXd multiply(const MatrixXd& a, const MatrixXd& b);
    VectorXd subtract(const VectorXd& a, const VectorXd& b);
    MatrixXd subtract(const MatrixXd& a, const MatrixXd& b);
    VectorXd add(const VectorXd& a, const VectorXd& b);
    MatrixXd add(const MatrixXd& a, const MatrixXd& b);
    double dot(const VectorXd& a, const VectorXd& b);
#endif
    void add_end_column(MatrixXd& a, const VectorXd& v);
    void remove_end_column(MatrixXd& a);
    void add_scale(const VectorXd& a, const VectorXd& x, const double& b, VectorXd& y);
    vector<size_t> argsort(const VectorXd& x);

};
#endif