#ifndef __LU_H
#define __LU_H

#include <vector>
#include <numeric>
using namespace std;
#if !defined(USING_EIGEN)
namespace lu
{
	void decompose(vector<vector<double>>& a, vector<size_t>& p);
	vector<double> solve(const vector<vector<double>>& a, const vector<size_t>& p, const vector<double>& x);
	vector<vector<double>> inverse(vector<vector<double>>& a);
	double determinant(const vector<vector<double>>& a, const vector<size_t>& p);
};
#endif
#endif