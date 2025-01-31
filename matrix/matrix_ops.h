#ifndef __MATRIX_OPS
#define __MATRIX_OPS

using namespace std;
#if defined(USING_EIGEN)
#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::VectorXd;
#else
typedef vector<double> VectorXd;
#endif
#include <vector>
namespace matrix_ops
{
#if !defined(USING_EIGEN)
    vector<vector<double>> identity(size_t n);
    vector<vector<double>> tril(const vector<vector<double>>& a, int k = 0);
    vector<double> diag(const vector<vector<double>>& a);
    vector<vector<double>> diag(const vector<double>& d);
    void initialise(vector<vector<double>>& a, const vector<double>& v);
    size_t num_columns(const vector<vector<double>>& a);
    size_t num_rows(const vector<vector<double>>& a);
    void add_end_column(vector<vector<double>>& a, const vector<double>& v);
    void remove_end_column(vector<vector<double>>& a);
    vector<double> scale(const vector<double>& x, double a);
    vector<vector<double>> scale(const vector<vector<double>>& x, double a);
    vector<vector<double>> copy(const vector<vector<double>>& a_);
    vector<vector<double>> vconcat(const vector<vector<double>>& a, const vector<vector<double>>& b);
    vector<vector<double>> hconcat(const vector<vector<double>>& a, const vector<vector<double>>& b);
    vector<double> get_col(const vector<vector<double>>& a, size_t col);
    vector<double> get_row(const vector<vector<double>>& a, size_t row);
    vector<vector<double>> transpose(const vector<vector<double>>& a);
    vector<double> multiply(const vector<vector<double>>& a, const vector<double>& x);
    vector<vector<double>> multiply(const vector<vector<double>>& a, const vector<vector<double>>& b);
    vector<double> subtract(const vector<double>& a, const vector<double>& b);
    vector<vector<double>> subtract(const vector<vector<double>>& a, const vector<vector<double>>& b);
    vector<double> add(const vector<double>& a, const vector<double>& b);
    vector<vector<double>> add(const vector<vector<double>>& a, const vector<vector<double>>& b);
    double dot(const vector<double>& a, const vector<double>& b);
#else
    void add_end_column(MatrixXd& a, const VectorXd& v);
    void remove_end_column(MatrixXd& a);
#endif
    void add_scale(const VectorXd& a, const VectorXd& x, const double& b, VectorXd& y);
    vector<size_t> argsort(const VectorXd& x);

};
#endif