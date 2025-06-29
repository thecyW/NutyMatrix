#include <iostream>
#include "Matrix.h"
#include <cassert>

int main()
{
	

	Matrix<float> A(2, 2);
	A(0, 0) = 0.2;  A(0, 1) = 0.7;
	A(1, 0) = 2;  A(1, 1) = 1;

	std::cout << A.determinant();
	
	Matrix<int> I = Matrix<int>::Identity(5);
	I.printMatrix();

	Matrix<int> O = Matrix<int>::Zero(2,3);
	O.printMatrix();

	return 0;

}