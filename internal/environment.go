package internal

import "github.com/jackkenney/evolve-rl/mathlib"

// Environment interface outlines abstract methods for an environment class.
type Environment interface {
	// How many episodes should be run?
	GetMaxEps() int
	// This function returns the dimension (length) of state vectors.
	GetStateDim() int
	// This function returns |\mathcal A|. Note that we are assuming that the action set is finite.
	GetNumActions() int
	// This function returns \gamma
	GetGamma() float64
	// This function applies action a, updating the state of the environment. It returns the reward that results from the state transition.
	Transition(a int, rng *mathlib.Random) float64
	// This function returns the current state of the environment
	GetState() []float64
	// Check if the current state is Terminal Absorbing State (TAS)
	InTAS() bool
	// This function resets the environment to start a new episode (it samples the state from the initial state distribution).
	NewEpisode(rng *mathlib.Random)
	// DeepCopy returns deep copy of struct
	DeepCopy(rng *mathlib.Random) Environment
}
