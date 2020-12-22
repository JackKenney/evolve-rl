package internal

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

var (
	agt Agent
)

func init() {
	source = rand.NewSource(0)
	rng = rand.New(source)
	agt = NewTabularBBO(23, 4, 0.9, 1, 1000)
	env = Gridworld{}
}

func TestUpdateBeforeNextAction(t *testing.T) {
	assert.False(t, agt.UpdateBeforeNextAction())
}

func TestGetAction(t *testing.T) {
	s := env.GetState()
	a := agt.GetAction(s, rng)
	assert.True(t, a == -1 || a == 0 || a == 1 || a == 2 || a == 3)
}

// func TestUpdateSARSA(t *testing.T) {
// 	s := env.GetState()
// 	a := agt.GetAction(s, rng)
// 	r := env.Transition(a, rng)
// 	sp := env.GetState()
// 	ap := agt.GetAction(sp, rng)
// 	agt.UpdateSARSA(s, a, r, sp, ap, rng)
// }
