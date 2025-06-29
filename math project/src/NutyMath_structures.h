#pragma once
#include "NutyMath.h"
struct Fraction
{
	int numerator;
	int denominator;

	// Constructor
	Fraction(int num, int denom)
		: numerator(num) , denominator(denom)
	{
		if (denom == 0)
		{
			throw std::invalid_argument(
				"Fraction error: denominator must be a non-zero integer."
			);
		}
		else if (denom < 0)
		{
			numerator *= -1;
			denominator *= -1; 
		}

		simplify();
	}

	void simplify()
	{
		int gcd = computedGcd(std::abs(numerator), denominator);
		numerator /= gcd;
		denominator /= gcd;
	}

	// Euclidean Algorithm
	static int computedGcd(int a, int b)
	{
		while (b != 0)
		{
			int temp = b;
			b = a % b;
			a = temp;
		}
		return a;
	}

	// Convert to double type
	double toDouble() const
	{
		return static_cast<double>(numerator) / denominator;
	}

	// Computes the reciprocal(multiplicative inverse) of the fraction
	Fraction reciprocal() const {
		if (numerator == 0) {
			throw std::invalid_argument(
				"Fraction reciprocal error: Cannot invert zero (0/" +
				std::to_string(denominator) + ")"
			);
		}
		return Fraction(denominator, numerator);
	}

	// Cal
	// 1.addition
	Fraction operator+(const Fraction& other) const
	{
		return Fraction(
			numerator * other.denominator + denominator * other.numerator,
			denominator * other.denominator
		);
	}
	// 2.multiplication
	Fraction operator*(const Fraction& other) const
	{
		return Fraction(
			numerator * other.numerator,
			denominator * other.denominator
		);
	}
	// 3.division
	Fraction operator/(const Fraction& other) const
	{
		if (other.numerator == 0)
		{
			throw std::invalid_argument(
				"Fraction division error: Division by zero (attempted to divide by fraction 0/" + std::to_string(other.denominator) + ")"
			);
		}

		return Fraction(
			numerator * other.denominator,
			denominator * other.numerator
		);
	}

	// Print
	void print() const
	{
		std:cout << std::to_string(numerator) << "/" << std::to_string(denominator);
	}
};