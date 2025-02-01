#include <algorithm>
#include <numeric>
#include "matrix_ops.h"

#if !defined(USING_EIGEN)
#include <stdexcept>

MatrixXd matrix_ops::identity(size_t n)
{
    MatrixXd a(n);
    for (size_t i = 0; i < n; ++i)
    {
        a[i].resize(n, 0.0);
        a[i][i] = 1.0;
    }
    return a;
}
MatrixXd matrix_ops::tril(const MatrixXd& a, int k)
{
    // returns lower triangle of a
    size_t n = a.size();
    size_t m = a[0].size();
    MatrixXd l(n, VectorXd(m, 0.0));
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            if (j <= i + k)
            {
                l[i][j] = a[i][j];
            }
        }
    }
    return l;
}
VectorXd matrix_ops::diag(const MatrixXd& a)
{
    size_t n = a.size();
    VectorXd d = VectorXd(n);
    for (size_t i = 0; i < n; ++i)
    {
        d[i] = a[i][i];
    }
    return d;
}
MatrixXd matrix_ops::diag(const VectorXd& d)
{
    size_t n = d.size();
    MatrixXd dm(n, VectorXd(n, 0.0));
    for (size_t i = 0; i < n; ++i)
    {
        dm[i][i] = d[i];
    }
    return dm;
}
void matrix_ops::initialise(MatrixXd& a, const VectorXd& v)
{
    if (!a.empty())
    {
        throw runtime_error("initialise matrix - a must be empty");
    }
    a = MatrixXd(v.size(), VectorXd(1));
    for (size_t i = 0; i < v.size(); ++i)
    {
        a[i][0] = v[i];
    }
}
void matrix_ops::add_end_column(MatrixXd& a, const VectorXd& v)
{
    if (a.empty())
    {
        initialise(a, v);
    }
    else
    {
        size_t num_rows = a.size();
        if (a.size() != v.size())
        {
            throw runtime_error("add_end_column: number of rows in a must be the same as size of v");
        }
        for (size_t i = 0; i < num_rows; ++i)
        {
            a[i].push_back(v[i]);
        }
    }
}
size_t matrix_ops::num_columns(const MatrixXd& a)
{
    return a.empty() ? 0 : a[0].size();
}
size_t matrix_ops::num_rows(const MatrixXd& a)
{
    return a.empty() ? 0 : a.size();
}
void matrix_ops::remove_end_column(MatrixXd& a)
{
    if (a.empty() || a[0].empty())
    {
        throw runtime_error("matrix a is empty or has no columns");
    }
    for (auto& row : a)
    {
        row.pop_back();
    }
}
VectorXd matrix_ops::scale(const VectorXd& x, double a)
{
    VectorXd y(x);
    transform(y.begin(), y.end(), y.begin(), [&a](auto& c) {return c * a; });
    return y;
}
MatrixXd matrix_ops::scale(const MatrixXd& x, double a)
{
    size_t n = x.size();
    MatrixXd s(n);
    for (size_t i = 0; i < n; ++i)
    {
        s[i] = scale(x[i], a);
    }
    return s;
}
MatrixXd matrix_ops::copy(const MatrixXd& a_)
{
    if (a_.empty())
    {
        throw runtime_error("copy: matrix a is empty");
    }
    size_t n = a_.size();
    MatrixXd a(n);
    for (size_t i = 0; i < n; ++i)
    {
        a[i].resize(a_[i].size());
        for (size_t j = 0; j < n; ++j)
        {
            a[i][j] = a_[i][j];
        }
    }
    return a;
}
MatrixXd matrix_ops::vconcat(const MatrixXd& a, const MatrixXd& b)
{
    MatrixXd ab = a;
    ab.insert(ab.end(), b.begin(), b.end());
    return ab;
}
MatrixXd matrix_ops::hconcat(const MatrixXd& a, const MatrixXd& b)
{
    size_t n = a.size();
    MatrixXd result(n);
    for (size_t i = 0; i < n; ++i)
    {
        result[i].insert(result[i].end(), a[i].begin(), a[i].end());
        result[i].insert(result[i].end(), b[i].begin(), b[i].end());
    }
    return result;
}
VectorXd matrix_ops::get_col(const MatrixXd& a, size_t col)
{
    if (a.empty() || col < 0 || col >= a[0].size())
    {
        throw out_of_range("Invalid column index!");
    }

    VectorXd column;
    for (const auto& row : a)
    {
        column.push_back(row[col]);
    }

    return column;
}
VectorXd matrix_ops::get_row(const MatrixXd& a, size_t row)
{
    if (row < 0 || row >= a.size())
    {
        throw out_of_range("invalid row index");
    }
    return a[row];
}
MatrixXd matrix_ops::transpose(const MatrixXd& a)
{
    size_t n = a.size(), m = a[0].size();
    MatrixXd t(m);
    for (size_t i = 0; i < m; ++i)
    {
        t[i].resize(n);
        for (size_t j = 0; j < n; ++j)
        {
            t[i][j] = a[j][i];
        }
    }
    return t;
}
VectorXd matrix_ops::multiply(const MatrixXd& a, const VectorXd& x)
{
    // mulliply a matrix of size (m, n) by a vector of size n
    size_t m = a.size();
    if (a.empty() || m == 0)
    {
        throw runtime_error("multiply - matrix a is empty");
    }
    size_t n = a[0].size();
    if (n != x.size())
    {
        throw runtime_error("multiply - number of cols in a must be the same as the number of rows in x");
    }
    VectorXd b(m);
    for (size_t i = 0; i < m; ++i)
    {
        b[i] = 0.0;
        for (size_t j = 0; j < n; ++j)
        {
            b[i] += a[i][j] * x[j];
        }
    }
    return b;
}
MatrixXd matrix_ops::multiply(const MatrixXd& a, const MatrixXd& b)
{
    // multiply two matrices
    size_t rows_a = a.size(), cols_a = a[0].size();
    size_t rows_b = b.size(), cols_b = b[0].size();
    if (rows_a != cols_b)
    {
        throw invalid_argument("Number of columns in a must equal the number of rows in b");
    }
    MatrixXd c(rows_a);
    for (size_t i = 0; i < rows_a; ++i)
    {
        c[i].resize(cols_b, 0.0);
        for (size_t j = 0; j < cols_b; ++j)
        {
            for (size_t k = 0; k < cols_a; ++k)
            {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return c;
}
VectorXd matrix_ops::subtract(const VectorXd& a, const VectorXd& b)
{
    size_t n = a.size();
    VectorXd c(n);
    for (size_t i = 0; i < n; ++i)
    {
        c[i] = a[i] - b[i];
    }
    return c;
}
MatrixXd matrix_ops::subtract(const MatrixXd& a, const MatrixXd& b)
{
    // subtract two matrices
    size_t m = a.size(); // number of rows
    size_t n = a[0].size(); // number of columns
    MatrixXd c(n);
    for (size_t i = 0; i < n; ++i)
    {
        c[i].resize(n);
        for (size_t j = 0; j < n; ++j)
        {
            c[i][j] = a[i][j] - b[i][j];
        }
    }
    return c;
}
VectorXd matrix_ops::add(const VectorXd& a, const VectorXd& b)
{
    size_t n = a.size();
    VectorXd c(n);
    for (size_t i = 0; i < n; ++i)
    {
        c[i] = a[i] + b[i];
    }
    return c;
}
MatrixXd matrix_ops::add(const MatrixXd& a, const MatrixXd& b)
{
    // add two matrices
    size_t m = a.size(); // number of rows
    size_t n = a[0].size(); // number of columns
    MatrixXd c(n);
    for (size_t i = 0; i < n; ++i)
    {
        c[i].resize(n);
        for (size_t j = 0; j < n; ++j)
        {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    return c;
}
double matrix_ops::dot(const VectorXd& a, const VectorXd& b)
{
    double sum = 0.0;
    size_t n = a.size();
    for (size_t i = 0; i < n; ++i)
    {
        sum += a[i] * b[i];
    }
    return sum;
}
#else
void matrix_ops::remove_end_column(MatrixXd& a)
{
    a.block(0, 0, a.rows(), a.cols() - 1) = a.leftCols(a.cols() - 1);
    a.conservativeResize(a.rows(), a.cols() - 1);
}
void matrix_ops::add_end_column(MatrixXd& a, const VectorXd& v)
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
#endif
void matrix_ops::add_scale(const VectorXd& a, const VectorXd& x, const double& b, VectorXd& y)
{
    transform(a.begin(), a.end(), x.begin(), y.begin(),
              [b](double ai, double xi) {
                  return ai + b * xi;
              });
}
vector<size_t> matrix_ops::argsort(const VectorXd& x)
{
    vector <size_t> idx(x.size());
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&x](size_t i, size_t j) { return x[i] < x[j]; });
    return idx;
}