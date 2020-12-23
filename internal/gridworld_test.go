package internal

import (
	"testing"

	"github.com/jackkenney/evolve-rl/mathlib"
	"github.com/stretchr/testify/assert"
)

var (
	grid *Gridworld
)

func init() {
	rng = mathlib.NewRandom(0)
	grid = NewGridworld(rng).(*Gridworld)
}

func TestConstructor(t *testing.T) {
	grid.NewEpisode(rng)
	assert.Equal(t, 0, grid.x, "grid.x not 0 initially")
	assert.Equal(t, 0, grid.y, "grid.y not 0 initially")
	assert.Equal(t, 0, grid.t, "grid.t not 0 initially")
	assert.Equal(t, false, grid.tas, "grid.tas not false initially")
}
func TestMaxEps(t *testing.T) {
	assert.Equal(t, 1000, grid.GetMaxEps())
}

func TestTransition(t *testing.T) {
	grid.NewEpisode(rng)
	grid.Transition(1, rng)
	s := grid.GetState()
	b := s[0] == 1.0 || s[1] == 1.0
	assert.True(t, b)
}

func TestTASLocation(t *testing.T) {
	grid = NewGridworld(rng).(*Gridworld)
	grid.x = 4
	grid.y = 4
	r := grid.Transition(0, rng)
	assert.Equal(t, 0.0, r, "wrong error returned from transition to TAS")
	assert.True(t, grid.tas, "grid.tas didn't change")
	assert.True(t, grid.InTAS(), "InTAS didn't change")
}

func TestTASTime(t *testing.T) {
	grid = NewGridworld(rng).(*Gridworld)
	grid.x = 0
	grid.y = 0
	grid.t = 99 // at t=100, Transition to TAS and get -100 for reward
	assert.NotPanics(t, func() {
		grid.GetState()
	})
	r := grid.Transition(0, rng)
	assert.Equal(t, -100.0, r, "wrong error returned from transition to TAS")
	assert.True(t, grid.tas, "grid.tas didn't change")
	assert.True(t, grid.InTAS(), "InTAS didn't change")
	assert.Panics(t, func() {
		grid.GetState()
	})
}
