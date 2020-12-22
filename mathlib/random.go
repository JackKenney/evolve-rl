package mathlib

import (
	"math/rand"
	"sync"
)

// Random manages a concurrent-safe random number generator.
type Random struct {
	mu     *sync.RWMutex
	source rand.Source
	rng    *rand.Rand
}

// NewRandom returns a new threadsafe random number generator.
func NewRandom(seed int64) *Random {
	r := Random{}
	r.mu = &sync.RWMutex{}
	r.source = rand.NewSource(seed)
	r.rng = rand.New(r.source)
	return &r
}

// Float64 generates threadsafe uniform random float from [0,1)
func (r *Random) Float64() float64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.rng.Float64()
}
