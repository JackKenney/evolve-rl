package internal

import (
	"math/rand"
)

// Agent is an interface that can be used to traverse and learn in an Environment.
type Agent interface {
	// UpdateBeforeNextAction makes an update to the agent's policy before selecting the next action.
	UpdateBeforeNextAction() bool
	// GetAction returns the action that the agent selects from the state.
	GetAction(s []float64, rng *rand.Rand) int
	// NewEpisode tells the agent that it is at the start of a new episode.
	NewEpisode()
	// Reset the agent entirely - to a blank slate prior to learning.
	Reset(rng *rand.Rand)
	// Update given a (s,a,r,s') tuple, if UpdateBeforeNextAction returns true.
	UpdateSARS(s []float64, a int, r float64, sPrime []float64, rng *rand.Rand)
	// Update given a (s,a,r,s',a') tuple, if UpdateBeforeNextAction returns false.
	UpdateSARSA(s []float64, a int, r float64, sPrime []float64, aPrime int, rng *rand.Rand)
	// LastUpdate lets the agent update/learn when sPrime would be the terminal absorbing state.
	LastUpdate(s []float64, a int, r float64, rng *rand.Rand)
	// // EpisodicUpdate the agent after N episodes (specified in constructor)
	// episodicUpdate(rng *rand.Rand)
	// // Clear agent's memory
	// wipeStatesActionsRewards()
	// DeepCopy returns deep copy of struct
	DeepCopy() Agent
}
