package mathlib

// Vector creates a slice with initial value.
func Vector(size int, initialValue float64) []float64 {
	vec := make([]float64, size)
	for i := 0; i < size; i++ {
		vec[i] = initialValue
	}
	return vec
}

// ZeroVec zeros out the values in vector vec
func ZeroVec(vec *[]float64) {
	for i := 0; i < len(*vec); i++ {
		(*vec)[i] = 0
	}
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

// Column returns the k'th column of the passed Matrix mat.
func Column(mat [][]float64, k int) []float64 {
	vec := make([]float64, len(mat))
	for i := 0; i < len(mat); i++ {
		vec[i] = mat[i][k]
	}
	return vec
}

// ZeroMat zeros out the values in matrix mat
func ZeroMat(mat *[][]float64) {
	for i := 0; i < len(*mat); i++ {
		for j := 0; j < len((*mat)[i]); j++ {
			(*mat)[i][j] = 0
		}
	}
}

// FromOneHot returns index of one hot and panics if vector not one hot
func FromOneHot(v []float64) int {
	state := 0
	for state = 0; state < len(v); state++ {
		if v[state] != 0 {
			break
		}
	}
	if state == len(v) {
		// If this happens, the s-vector was all zeros
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
