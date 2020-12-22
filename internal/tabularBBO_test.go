package internal

import (
	"testing"

	"github.com/jackkenney/evolve-rl/mathlib"
	"github.com/stretchr/testify/assert"
)

var (
	agt Agent
)

func init() {
	rng = mathlib.NewRandom(0)
	agt = NewTabularBBO(23, 4, 0.9, 1, 1000).(Agent)
	env = NewGridworld(rng)
}

func TestUpdateBeforeNextAction(t *testing.T) {
	assert.False(t, agt.UpdateBeforeNextAction())
}

func TestGetAction(t *testing.T) {
	s := env.GetState()
	a := agt.GetAction(s, rng)
	assert.True(t, a == -1 || a == 0 || a == 1 || a == 2 || a == 3)
}

func TestUpdateSARSA(t *testing.T) {
	s := env.GetState()
	a := agt.GetAction(s, rng)
	r := env.Transition(a, rng)
	sp := env.GetState()
	ap := agt.GetAction(sp, rng)
	oldT := agt.(*TabularBBO).t
	agt.UpdateSARSA(s, a, r, sp, ap, rng)
	newT := agt.(*TabularBBO).t
	assert.NotEqual(t, oldT, newT, "time did not update")
}

func TestNLimit(t *testing.T) {
	agt.(*TabularBBO).epCount = agt.(*TabularBBO).N
	s := env.GetState()
	a := agt.GetAction(s, rng)
	r := env.Transition(a, rng)
	sp := env.GetState()
	ap := agt.GetAction(sp, rng)
	agt.UpdateSARSA(s, a, r, sp, ap, rng)
}
