package internal

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

var (
	grid Gridworld
)

func init() {
	source = rand.NewSource(0)
	rng = rand.New(source)
	grid = NewGridworld(rng).(Gridworld)
}

func TestConstructor(t *testing.T) {
	assert.Equal(t, 0, grid.x)
	assert.Equal(t, 0, grid.y)
	assert.Equal(t, 0, grid.t)
	assert.Equal(t, false, grid.tas)
}
func TestMaxEps(t *testing.T) {
	assert.Equal(t, 1000, grid.GetMaxEps())
}

func TestTransition(t *testing.T) {
	grid.Transition(1, rng)
	s := grid.GetState()
	b := s[0] == 1.0 || s[1] == 1.0
	assert.True(t, b)
}
