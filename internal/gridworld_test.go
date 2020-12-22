package internal

import (
	"fmt"
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
func TestTAS(t *testing.T) {
	grid = NewGridworld(rng).(Gridworld)
	grid.x = 4
	grid.y = 4
	r := grid.Transition(0, rng)
	fmt.Println(r)
	assert.Equal(t, 1, grid.t, "time did not increment")
	assert.True(t, grid.tas, "grid.tas not correct")
	assert.True(t, grid.InTAS(), "InTAS does not work")
}
