#pragma once
#include "NutyMath.h"
#include "NutyMath_structures.h"
template <typename T>

class Matrix {
private:
	std::vector<std::vector<T>> _data;
	size_t _rows, _cols;

public:

	// Constructor
	Matrix() : _rows(0), _cols(0), _data() {}

	Matrix(size_t rows, size_t cols)
		: _rows(rows), _cols(cols)
	{
		_data.resize(rows);
		for (std::vector<T>& vec : _data)
		{
			vec.resize(cols);
		}
	}

	Matrix(size_t rows, size_t cols, const T& initial_value)
		: _rows(rows), _cols(cols)
	{
		_data.resize(rows);
		for (std::vector<T>& vec : _data)
		{
			vec.resize(cols);
			for (T& val : vec)
			{
				val = initial_value;
			}
		}
	}

	Matrix(std::vector<std::vector<T>>& data)
	{
		_data = data;
		_rows = data.size();
		_cols = data[0].size();
	}

	// Copy constructor   Assignment operator
	Matrix(const Matrix& other)
		: _rows(other._rows), _cols(other._cols), _data(other._rows, std::vector<T>(other._cols))
	{
		for (size_t i = 0; i < _rows; ++i) {
			for (size_t j = 0; j < _cols; ++j) {
				_data[i][j] = other(i,j); 
			}
		}
	}

	Matrix& operator=(const Matrix& other)
	{
		if (this != &other)
		{
			std::vector<std::vector<T>>().swap(_data);
		}
		_rows = other.getRows();
		_cols = other.getCols();
		_data.resize(_rows);

		int i = 0;
		for (std::vector<T>& vec : _data)
		{
			vec.resize(_cols);
			for (int j = 0; j < other.getCols(); j++)
			{
				vec[j] = other(i, j);
			}
		}

		return *this;
	}

	// Move constructor
	Matrix(Matrix&& other) noexcept
		: _rows(other._rows), _cols(other._cols), _data(std::move(other._data))
	{
		other._rows = 0;
		other._cols = 0;
	}

	// Move assignment operator
	Matrix& operator=(Matrix&& other) noexcept 
	{
		if (this != &other) {
			_rows = other._rows;
			_cols = other._cols;
			_data = std::move(other._data);
			other._rows = 0;
			other._cols = 0;
		}
		return *this;
	}

	// Mixed-type conversion
	template <typename U>
	Matrix(const Matrix<U>& other) 
		: _rows(other.getRows()), _cols(other.getCols()) 
	{
		_data.resize(_rows);
		for (int i = 0; i < _rows; ++i) {
			_data[i].resize(_cols);
			for (int j = 0; j < _cols; ++j) {
				_data[i][j] = static_cast<T>(other(i, j));
			}
		}
	}

	template <typename U>
	Matrix<U> toRounded() const {
		Matrix<U> result(_rows, _cols);
		for (int i = 0; i < _rows; ++i) {
			for (int j = 0; j < _cols; ++j) {
				result(i, j) = static_cast<U>(std::round((*this)(i, j)));
			}
		}
		return result;
	}

	// Get the element    0-based
	T& operator()(size_t row, size_t col)
	{
		if (row >= this -> getRows() || col >= this -> getCols())
		{
			throw std::out_of_range("Matrix access out of bounds: (" +
				std::to_string(row) + ", " + std::to_string(col) + ") for matrix of size " +
				std::to_string(this->_rows) + "x" + std::to_string(this->_cols)
			);
		}
		return _data[row][col];
	}

	const T& operator()(size_t row, size_t col) const
	{
		if (row >= this->getRows() || col >= this->getCols())
		{
			throw std::out_of_range("Matrix access out of bounds: (" +
				std::to_string(row) + ", " + std::to_string(col) + ") for matrix of size " +
				std::to_string(this->_rows) + "x" + std::to_string(this->_cols)
			);
		}
		return _data[row][col];
	}

	/********    Get basic info about Matrix     ********/ 
	constexpr size_t getRows() const noexcept { return _rows; }
	constexpr size_t getCols() const noexcept { return _cols; }

	// Calculate the determinant of matrix
	T determinant() const;

	// Calculate the trace of matrix ()
	T trace() const
	{
		if (_rows != _cols)
		{
			throw std::invalid_argument(
				"Matrix trace computation failed: Trace is only defined for square matrices. Current matrix dimensions: " + std::to_string(_rows) + "x" + std::to_string(_cols) + ")"
			);
		}

		T trace = T(0);

		for (int i = 0 ; i < _cols ; ++ i)
		{
			trace += _data[i][i];
		}
		return trace;
	}

	/*// Calculate matrix norm ()
	template <typename U>
	U norm() const
	{

	}*/

	// Calculate
	Matrix operator+(const Matrix& other)
	{
		if (this -> getRows() != other.getRows() || this -> getCols() != other.getCols())
		{
			throw std::invalid_argument("Matrix addition failed: Dimension mismatch (" +
				std::to_string(_rows) + "x" + std::to_string(_cols) + ") vs (" +
				std::to_string(other._rows) + "x" + std::to_string(other._cols) + ")"
			);
		}

		Matrix<T> res(this -> getRows(), other.getCols());
		for (int i = 0 ; i < res.getRows() ; i ++)
		{
			for (int j = 0 ; j < res.getCols(); j ++)
			{
				res(i, j) = (*this)(i , j) + other(i , j);
			}
		}

		return res;
	}

	Matrix operator*(const Matrix& other)
	{
		if (this -> getCols() != other.getRows())
		{
			throw std::invalid_argument("Matrix multiplication failed: A.cols(" + std::to_string(this->_cols) + ") != B.rows(" + std::to_string(other._rows) + ")"
			);
		}

		Matrix<T> res(this -> getRows(), other.getCols());
		for (int i = 0; i < res.getRows(); i ++)
		{
			for (int j = 0 ; j < res.getCols(); j ++)
			{
				res(i, j) = 0;
				for (int k = 0; k < this->getCols(); k ++)
				{
					res(i, j) += (*this)(i, k) * other(k, j);
				}
			}
		}

		return res;
	}

	// Transpose
	Matrix transpose() const
	{
		// Create result matrix with swapped dimensions
		Matrix<T> result(_cols, _rows);

		// Optimized element-wise copying
		for (int i = 0; i < _rows; ++i) {
			for (int j = 0; j < _cols; ++j) {
				// No bounds checking here for performance
				// Safe because we control the loop limits
				result._data[j][i] = _data[i][j];
			}
		}

		return result;
	}

	void transposeInPlace() {
		if (_rows != _cols) {
			throw std::invalid_argument(
				"In-place transpose failed: Matrix is not square (" +
				std::to_string(_rows) + "x" + std::to_string(_cols) + ")"
			);
		}

		for (int i = 0; i < _rows; ++i) {
			for (int j = i + 1; j < _cols; ++j) {
				std::swap(_data[i][j], _data[j][i]);
			}
		}
	}

	// Swap
	void matrixSwapR(size_t r1, size_t r2)
	{
		std::swap(_data[r1], _data[r2]);
	}

	void matrixSwapC(size_t c1, size_t c2)
	{
		for (int row = 0; row < this -> getRows(); ++row)
		{
			T proc = (*this)(row, c1);
			(*this)(row, c1) = (*this)(row, c2);
			(*this)(row, c2) = proc;
		}
	}

	// Inverse
	Matrix inverse() const
	{
		// Check if matrix is square
		if (_rows != _cols)
		{
			throw std::invalid_argument("Matrix inversion failed: Matrix is not square (" + 
				std::to_string(_rows) + "x" + std::to_string(_cols) + ")"
			);
		}

		Matrix mat = *this; // Create working copy
		Matrix inverse(_rows, _cols); // Create identity matrix

		for (int i = 0; i < _rows; i++)
		{
			inverse(i, i) = T(1);
		}
		//Gauss-Jordan Elimination
		for (int col = 0; col < mat.getCols(); ++col)
		{
			// Partial pivoting: Find row with maximum absolute value in current column
			int max_row = col;
			for (int row = col + 1; row < mat.getRows(); ++row)
			{
				if (std::abs(mat(row , col) > std::abs(mat(max_row , col))))
				{
					max_row = row;
				}
			}

			if (max_row != col)
			{
				mat.matrixSwapR(max_row, col);
				inverse.matrixSwapR(max_row , col);
			}

			// Check for singularity
			if (mat(col, col) < 1e-10)
			{
				throw std::runtime_error("Matrix is singular (non-invertible)");
			}

			// Normalization step: Make diagonal element 1 by dividing entire row
			T pivot = mat(col, col);
			for (int j = 0; j < mat.getCols(); j++)
			{
				mat(col, j) /= pivot;
				inverse(col, j) /= pivot;
			}

			// Elimination step: Make all other elements in current column zero
			for (int i = 0; i < mat.getCols(); i++)
			{
				if (i != col && std::abs(mat(i, col)) > 0)
				{
					T factor = mat(i, col);

					for (int j = 0; j < mat.getCols(); j++)
					{
						mat(i, j) -= mat(col, j) * factor;
						inverse(i, j) -= inverse(col, j) * factor;
					}
				}
			}

		}

		return inverse;
	}
	
	// Inverse as double
	Matrix<double> inverseAsDouble() const 
	{
		if (_rows != _cols) {
			throw std::invalid_argument("Matrix inversion failed: Matrix is not square (" +
				std::to_string(_rows) + "x" + std::to_string(_cols) + ")"
			);
		}

		// Convert to double
		Matrix<double> mat_d(_rows, _cols);
		for (size_t i = 0; i < _rows; ++i) {
			for (size_t j = 0; j < _cols; ++j) {
				mat_d(i, j) = static_cast<double>((*this)(i, j));
			}
		}

		// Compute inverse using double-precision arithmetic
		return mat_d.inverse();
	}

	// Print matrix
	void printMatrix() const
	{
		std::cout << std::endl;
		for (int i = 0; i <  _rows; i ++)
		{
			for (int j = 0 ; j < _cols; j ++)
			{
				std::cout << (*this)(i, j) << ' ';
			}
			std::cout << std::endl;
		}
	}

	/*****    Factory method    *****/
	// 1.Create identity matrix
	static Matrix Identity(size_t n) 
	{
		Matrix mat(n, n);
		for (size_t i = 0; i < n; ++i) mat(i, i) = T(1);
		return mat;
	}
	// 2.Create zero matrix
	static Matrix Zero(size_t rows, size_t cols)
	{
		return Matrix(rows, cols, T(0));
	}
	// 3.Create diagonal matrix
	static Matrix Diagonal(const std::vector<T>& value)
	{
		Matrix mat(value.size(), value.size());

		for (int i = 0 ; i <  value.size(); i ++)
		{
			mat(i, i) = value[i];
		}

		return mat;
	}
};


// Calculate the determinant of matrix (LU decomposition)
template<typename T>
T Matrix<T>::determinant() const
{
	// Check if matrix is square
	if (_rows != _cols)
	{
		throw std::invalid_argument(
			"Determinant calculation failed: Matrix is not square (" +
			std::to_string(_rows) + "x" + std::to_string(_cols) + ")"
		);
	}

	// Direct formula for 1x1 matrix
	if (_rows == 1)
	{
		return _data[0][0];
	}
	// Direct formula for 2x2 matrix (more efficient than general algorithm)
	else if (_rows == 2)
	{
		return _data[0][0] * _data[1][1] - _data[0][1] * _data[1][0];
	}
	// Direct formula for 3x3 matrix (Sarrus' rule - more efficient)
	else if (_rows == 3)
	{
		return _data[0][0] * (_data[1][1] * _data[2][2] - _data[1][2] * _data[2][1]) -
			_data[0][1] * (_data[1][0] * _data[2][2] - _data[1][2] * _data[2][0]) +
			_data[0][2] * (_data[1][0] * _data[2][1] - _data[1][1] * _data[2][0]);
	}

	Matrix<T> temp = *this;// Create working copy

	// Initialize determinant as multiplicative identity (1)
	T det = T(1);

	// Perform Gaussian elimination with partial pivoting
	for (int col = 0; col < temp.getCols(); col++)
	{
		int max_row = col;

		for (int row = col + 1; row < temp.getRows(); row++)
		{
			if (std::abs(temp(row, col)) > std::abs(temp(max_row, col)))
			{
				max_row = row;
			}
		}

		// If maximum element is zero, matrix is singular (determinant = 0)
		if (temp(max_row, col) == T(0))
		{
			return T(0);
		}

		// Swap rows if necessary to get the largest pivot element
		if (max_row != col)
		{
			temp.matrixSwapR(max_row, col);
			det = -det;
		}

		// Multiply determinant by the diagonal element (pivot)
		det *= temp(col, col);

		// Perform elimination for all rows below the current one
		for (int row = col + 1; row < temp.getRows(); row++)
		{
			T factor = temp(row, col) / temp(col, col);

			for (int c = col; c < temp.getCols(); c++)
			{
				temp(row, c) -= temp(col, c) * factor;
			}
		}
	}

	return det;

}

template<>
int Matrix<int>::determinant() const
{
	// Check if matrix is square
	if (_rows != _cols)
	{
		throw std::invalid_argument(
			"Determinant calculation failed: Matrix is not square (" +
			std::to_string(_rows) + "x" + std::to_string(_cols) + ")"
		);
	}
	// Direct formula for 1x1 matrix
	if (_rows == 1)
	{
		return _data[0][0];
	}
	// Direct formula for 2x2 matrix (more efficient than general algorithm)
	else if (_rows == 2)
	{
		return _data[0][0] * _data[1][1] - _data[0][1] * _data[1][0];
	}
	// Direct formula for 3x3 matrix (Sarrus' rule - more efficient)
	else if (_rows == 3)
	{
		return _data[0][0] * (_data[1][1] * _data[2][2] - _data[1][2] * _data[2][1]) -
			_data[0][1] * (_data[1][0] * _data[2][2] - _data[1][2] * _data[2][0]) +
			_data[0][2] * (_data[1][0] * _data[2][1] - _data[1][1] * _data[2][0]);
	}

	Matrix<int> temp = *this; // Create working copy
	int det = 1;
	int sign = 1;

	// Perform Gaussian elimination with partial pivoting
	for (int col = 0; col < temp.getCols(); col++)
	{
		int max_row = col;

		for (int row = col + 1; row < temp.getRows(); row++)
		{
			if (std::abs(temp(row, col)) > std::abs(temp(max_row, col)))
			{
				max_row = row;
			}
		}

		// If maximum element is zero, matrix is singular (determinant = 0)
		if (temp(max_row, col) == 0)
		{
			return 0;
		}

		// Swap rows if necessary to get the largest pivot element
		if (max_row != col)
		{
			temp.matrixSwapR(max_row, col);
			sign = -sign;
		}

		// Multiply determinant by the diagonal element (pivot)
		det *= temp(col, col);

		// Perform elimination for all rows below the current one
		for (int row = col + 1; row < temp.getRows(); row++)
		{
			Fraction factor(temp(row, col), temp(col, col));

			for (int c = col; c < temp.getCols(); c++)
			{
				temp(row, c) = (temp(col, c) * factor.denominator - factor.numerator) / factor.denominator;
			}
		}
	}

	return det * sign;

}