package internal

import (
	"math/rand"

	"github.com/stretchr/testify/assert"
)

var (
	source rand.Source
	rng    *rand.Rand
	a      assert.Assertions
	env    Environment
)

func init() {
	source = rand.NewSource(0)
	rng = rand.New(source)
}
