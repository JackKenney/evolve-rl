package internal

import (
	"math"

	"github.com/jackkenney/evolve-rl/mathlib"
)

// Sarsa learning agent using black box optimization
type Sarsa struct {
	numStates  int     // How many discrete states?
	numActions int     // How many discrete actions?
	gamma      float64 // Discount parameter

	theta           [][]float64 // Q (Action-Value) Function
	alpha           float64
	optimisticValue float64
}

// NewSarsa returns an initialized Sarsa object.
func NewSarsa(stateDim int, numActions int, gamma float64, alpha float64, optimisticValue float64) Agent {
	agt := Sarsa{}
	agt.alpha = alpha

	agt.numStates = stateDim
	agt.numActions = numActions
	agt.gamma = gamma
	agt.optimisticValue = optimisticValue

	agt.theta = mathlib.Matrix(agt.numStates, agt.numActions, optimisticValue)

	return &agt
}

// UpdateBeforeNextAction makes an update to the agent's policy before selecting the next action.
func (agt *Sarsa) UpdateBeforeNextAction() bool {
	return false
}

// EpisodicAgent returns whether the agent makes end of episode updates
func (agt *Sarsa) EpisodicAgent() bool {
	return false
}

// GetAction return softmax selected action
func (agt *Sarsa) GetAction(s []float64, rng *mathlib.Random) int {
	// Convert the one-hot state into an integer from 0 - (numStates-1)
	state := mathlib.FromOneHot(s)
	// Get the action probabilities from theta, using softmax action selection.
	actionProbabilities := make([]float64, len(agt.theta[state]))
	copy(agt.theta[state], actionProbabilities)

	denominator := 0.0
	for a := 0; a < len(actionProbabilities); a++ {
		actionProbabilities[a] = math.Exp(actionProbabilities[a])
		denominator += actionProbabilities[a]
	}
	for a := 0; a < len(actionProbabilities); a++ {
		actionProbabilities[a] /= denominator
	}
	// Select random action from softmax
	temp := rng.Float64()
	sum := 0.0
	for a := 0; a < agt.numActions; a++ {
		sum += actionProbabilities[a]
		if temp <= sum {
			return a // The function will return 'a'. This stops the for loop and returns from the function.
		}
	}
	return agt.numActions - 1 // Rounding error
}

// NewEpisode tells the agent that it is at the start of a new episode.
func (agt *Sarsa) NewEpisode() {
	// Nothing to do at episode threshold for sarsa
}

// Reset the agent entirely - to a blank slate prior to learning
func (agt *Sarsa) Reset(rng *mathlib.Random) {
	mathlib.ResetMat(&agt.theta, agt.optimisticValue)
}

// UpdateSARS is unimplemented for this class.
func (agt *Sarsa) UpdateSARS(s []float64, a int, r float64, sPrime []float64, rng *mathlib.Random) {
	// Shouldn't be using this function
	panic("UpdateSARS is not implemented for Sarsa.")
}

// UpdateSARSA - given a (s,a,r,s',a') tuple
func (agt *Sarsa) UpdateSARSA(s []float64, a int, r float64, sPrime []float64, aPrime int, rng *mathlib.Random) {
	sIdx := mathlib.FromOneHot(s)
	sPrimeIdx := mathlib.FromOneHot(sPrime)
	tdError := r + agt.gamma*agt.theta[sPrimeIdx][aPrime] - agt.theta[sIdx][a]
	agt.theta[sIdx][a] += agt.alpha * tdError
}

// LastUpdate lets the agent update/learn when sPrime would be the terminal absorbing state.
func (agt *Sarsa) LastUpdate(s []float64, a int, r float64, rng *mathlib.Random) {
	state := mathlib.FromOneHot(s)
	tdError := r - agt.theta[state][a]
	agt.theta[state][a] += agt.alpha * tdError
}
