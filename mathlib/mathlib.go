package mathlib

import (
	"math"
)

// Vector creates a slice with initial value.
func Vector(size int, initialValue float64) []float64 {
	vec := make([]float64, size)
	for i := 0; i < size; i++ {
		vec[i] = initialValue
	}
	return vec
}

// Column returns the k'th column of the passed Matrix mat.
func Column(mat [][]float64, k int) []float64 {
	vec := make([]float64, len(mat))
	for i := 0; i < len(mat); i++ {
		vec[i] = mat[i][k]
	}
	return vec
}

// Matrix returns an initialized matrix of [rows, cols] by initialValue.
func Matrix(rows int, cols int, initialValue float64) [][]float64 {
	mat := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		mat[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			mat[i][j] = initialValue
		}
	}
	return mat
}

// FromOneHot returns index of one hot and panics if vector not one hot
func FromOneHot(v []float64) int {
	state := 0
	for state = 0; state < len(v); state++ {
		if v[state] != 0 {
			break
		}
	}
	if state == len(v) { // If this happens, the s-vector was all zeros
		panic("state vector was all zeros")
	}
	return state
}

// ToOneHot returns index of one hot and panics if vector not one hot
func ToOneHot(idx int, capacity int) []float64 {
	if idx >= capacity {
		panic("Cannot index past capacity. Arguments incorrect.")
	}
	v := make([]float64, capacity)
	v[idx] = 1
	return v
}

// Mean returns the mean value in this slices of float64 values.
func Mean(v []float64) float64 {
	total := 0.0
	for _, number := range v {
		total = total + number
	}
	average := total / float64(len(v)) // len  function return array size
	return average
}

// StdError calculate and return the standard error for this slice.
func StdError(v []float64) float64 {
	// The most common object types are int (integer), double (double precision floating point), bool (Boolean), VectorXd (vector), MatrixXd (matrix), and vector<type>. We'll talk about vector<type> later.
	sampleMean := Mean(v) // First, get the mean of the vector
	temp := 0.0           // Create a floating point (double precision) equal to zero
	// Below the (int) term means "cast the next thing into an "int" type. v.size() actually returns a long integer. C++ will automatically cast it to an int to compare to i, but your compiler might give you a warning that you're comparing two different integer types. The explicit casting to an "int" here avoids that warning.
	for _, val := range v { // This is a basic for loop. The variable i is initialized to zero at the start, it runs as long as i < (int)v.size(), and at the end of every iteration of the loop it calls i++ (i = i + 1).
		temp += (val - sampleMean) * (val - sampleMean) // temp += foo; means the same thing as temp = temp + foo;
	}
	return math.Sqrt(temp/float64(len(v)-1.0)) / math.Sqrt(float64(len(v))) // Return the standard error. The returned object must match the return type in the function delaration.
}
