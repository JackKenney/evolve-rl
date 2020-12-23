package internal

import (
	"math"

	"github.com/jackkenney/evolve-rl/mathlib"
)

// REINFORCE learning agent using black box optimization
type REINFORCE struct {
	numStates  int     // How many discrete states?
	numActions int     // How many discrete actions?
	gamma      float64 // Discount parameter

	ep *EpisodeTracker // tracks history for use by episodic agent

	theta [][]float64 // The current best policy we have found
	alpha float64
}

// NewREINFORCE returns an initialized REINFORCE object.
func NewREINFORCE(stateDim int, numActions int, gamma float64, alpha float64) Agent {
	agt := REINFORCE{}
	agt.alpha = alpha

	agt.ep = NewEpisodeTracker(1)
	agt.numStates = stateDim
	agt.numActions = numActions
	agt.gamma = gamma

	initialValue := 0.0
	agt.theta = mathlib.Matrix(agt.numStates, agt.numActions, initialValue)

	return &agt
}

// UpdateBeforeNextAction makes an update to the agent's policy before selecting the next action.
func (agt *REINFORCE) UpdateBeforeNextAction() bool {
	return false
}

// EpisodicAgent returns whether the agent makes end of episode updates
func (agt *REINFORCE) EpisodicAgent() bool {
	return true
}

// GetAction returns the action that the agent selects from the state.
func (agt *REINFORCE) GetAction(s []float64, rng *mathlib.Random) int {
	// Convert the one-hot state into an integer from 0 - (numStates-1)
	state := mathlib.FromOneHot(s)
	// Get the action probabilities from theta, using softmax action selection.
	actionProbabilities := agt.theta[state]
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
func (agt *REINFORCE) NewEpisode() {}

// Reset the agent entirely - to a blank slate prior to learning
func (agt *REINFORCE) Reset(rng *mathlib.Random) {
	mathlib.ZeroMat(&agt.theta)
	agt.ep.Wipe()
}

// UpdateSARS is unimplemented for this class.
func (agt *REINFORCE) UpdateSARS(s []float64, a int, r float64, sPrime []float64, rng *mathlib.Random) {
	// Shouldn't be using this function
	panic("UpdateSARS is not implemented for REINFORCE.")
}

// UpdateSARSA - given a (s,a,r,s',a') tuple
func (agt *REINFORCE) UpdateSARSA(s []float64, a int, r float64, sPrime []float64, aPrime int, rng *mathlib.Random) {
	agt.ep.Update(s, a, r, s)
}

// LastUpdate lets the agent update/learn when sPrime would be the terminal absorbing state.
func (agt *REINFORCE) LastUpdate(s []float64, a int, r float64, rng *mathlib.Random) {
	// If ready to update, update and wipe the states, actions, and rewards.
	if agt.ep.LastUpdate(s, a, r) {
		agt.episodeLimitReached(rng)
	}
}

func (agt *REINFORCE) episodeLimitReached(rng *mathlib.Random) {
	agt.episodicUpdate(rng)
	agt.ep.Wipe()
}

// EpisodicUpdate the agent after N episodes (specified in constructor)
func (agt *REINFORCE) episodicUpdate(rng *mathlib.Random) {
	// There is only one episode stored
	// Start by computing the unbiased estimate of the policy gradient
	gradientEstimate := mathlib.Matrix(agt.numStates, agt.numActions, 0.0)

	var s, a int
	L := len(agt.ep.rewards[0])
	G := 0.0

	var piS, state []float64

	for t := 0; t < L; t++ {
		state = agt.ep.states[0][t]
		s = mathlib.FromOneHot(state)
		a = agt.ep.actions[0][t]

		G = 0.0
		for k := t; k < L; k++ {
			G += math.Pow(agt.gamma, float64(k-t)) * agt.ep.rewards[0][k]
		}
		piS = mathlib.ExpVec(agt.theta[s])
		piS = mathlib.ScalarDivideVec(piS, mathlib.Sum(piS))

		gradientEstimate[s][a] += G * (1.0 - piS[a])
		for aPrime := 0; aPrime < agt.numActions; aPrime++ {
			if aPrime != a {
				gradientEstimate[s][aPrime] += G * (-piS[aPrime])
			}
		}
	}
	// Note: Drop the gamma^t term that is usually dropped in actual implementations!

	// Perform the actual update.
	scaledGradientEstimate := mathlib.ScalarMultiplyMat(gradientEstimate, agt.alpha)
	agt.theta = mathlib.AddMatrix(agt.theta, scaledGradientEstimate)
}
