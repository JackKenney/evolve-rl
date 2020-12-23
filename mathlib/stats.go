package mathlib

import "math"

// Sum returns the mean value in this slices of float64 values.
func Sum(v []float64) float64 {
	total := 0.0
	for _, number := range v {
		total = total + number
	}
	return total
}

// Mean returns the mean value in this slices of float64 values.
func Mean(v []float64) float64 {
	total := Sum(v)
	return total / float64(len(v))
}

// StdError calculate and return the standard error for this slice.
func StdError(v []float64) float64 {
	// The most common object types are int (integer), double (double precision floating point), bool (Boolean), VectorXd (vector), MatrixXd (matrix), and vector<type>. We'll talk about vector<type> later.
	sampleMean := Mean(v) // First, get the mean of the vector
	temp := 0.0           // Create a floating point (double precision) equal to zero
	// Below the (int) term means "cast the next thing into an "int" type. v.size() actually returns a long integer. C++ will automatically cast it to an int to compare to i, but your compiler might give you a warning that you're comparing two different integer types. The explicit casting to an "int" here avoids that warning.
	// This is a basic for loop. The variable i is initialized to zero at the start, it runs as long as i < (int)v.size(), and at the end of every iteration of the loop it calls i++ (i = i + 1).
	for _, val := range v {
		temp += (val - sampleMean) * (val - sampleMean) // temp += foo; means the same thing as temp = temp + foo;
	}
	return math.Sqrt(temp/float64(len(v)-1.0)) / math.Sqrt(float64(len(v))) // Return the standard error. The returned object must match the return type in the function delaration.
}
