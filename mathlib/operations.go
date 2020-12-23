package mathlib

import "math"

// ScalarMultiplyVec divides the passed vector by the passed scalar
func ScalarMultiplyVec(v []float64, c float64) []float64 {
	for i := 0; i < len(v); i++ {
		v[i] *= c
	}
	return v
}

// ScalarDivideVec divides the passed vector by the passed scalar
func ScalarDivideVec(v []float64, c float64) []float64 {
	for i := 0; i < len(v); i++ {
		v[i] /= c
	}
	return v
}

// ExpVec exponentiates each element in the vector and returns a copy of it.
func ExpVec(v []float64) []float64 {
	for i := 0; i < len(v); i++ {
		v[i] = math.Exp(v[i])
	}
	return v
}

// AddMatrix adds matrix a and b together element-wise and returns both
func AddMatrix(a [][]float64, b [][]float64) [][]float64 {
	if len(a) != len(b) {
		panic("a and b have different number of rows")
	}
	for i := 0; i < len(a); i++ {
		if len(a[i]) != len(b[i]) {
			panic("a and b have different number of columns")
		}
		for j := 0; j < len(a[i]); i++ {
			a[i][j] *= b[i][j]
		}
	}
	return a
}

// ScalarMultiplyMat divides the passed vector by the passed scalar
func ScalarMultiplyMat(mat [][]float64, c float64) [][]float64 {
	for i := 0; i < len(mat); i++ {
		for j := 0; j < len(mat[i]); i++ {
			mat[i][j] *= c
		}
	}
	return mat
}
