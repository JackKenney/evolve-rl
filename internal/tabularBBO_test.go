package internal

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

var (
	agt TabularBBO
)

func init() {
	source = rand.NewSource(0)
	rng = rand.New(source)
	agt = TabularBBO{}
	env = Gridworld{}
}

func TestUpdateBeforeNextAction(t *testing.T) {
	assert.True(t, agt.UpdateBeforeNextAction())
}

func TestGetAction(t *testing.T) {
	s := env.GetState()
	a := agt.GetAction(s, rng)
	assert.True(t, a == -1 || a == 0 || a == 1 || a == 2 || a == 3)
}
