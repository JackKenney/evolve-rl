package internal

import (
	"github.com/jackkenney/evolve-rl/mathlib"
	"github.com/stretchr/testify/assert"
)

var (
	rng *mathlib.Random
	a   assert.Assertions
	env Environment
)

func init() {
	rng = mathlib.NewRandom(0)
}
