package internal

import (
	"math/rand"
)

// EpisodicAgent is an agent that learns at the end of the episode (Monte Carlo based, usually).
type EpisodicAgent interface {
	UpdateBeforeNextAction() bool
	GetAction(s []float64, rng *rand.Rand) int
	// Tell the agent that it is at the start of a new episode
	NewEpisode()
	// Reset the agent entirely - to a blank slate prior to learning
	Reset(rng *rand.Rand)
	// Update given a (s,a,r,s') tuple
	Update(s []float64, a int, r float64, sPrime []float64, rng *rand.Rand)
	// Let the agent update/learn when sPrime would be the terminal absorbing state
	LastUpdate(s []float64, a int, r float64, rng *rand.Rand)
	// EpisodicUpdate the agent after N episodes (specified in constructor)
	EpisodicUpdate(rng *rand.Rand)
	// Clear agent's memory
	wipeStatesActionsRewards()
}
