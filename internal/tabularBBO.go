package internal

import (
	"math"

	"github.com/jackkenney/evolve-rl/mathlib"
)

// TabularBBO learning agent using black box optimization
type TabularBBO struct {
	numStates  int     // How many discrete states?
	numActions int     // How many discrete actions?
	gamma      float64 // Discount parameter

	ep *EpisodeTracker // tracks history for use by episodic agent

	curTheta     [][]float64 // The current best policy we have found
	curThetaJHat float64     // $\hat J(\theta_\text{cur})$ in LaTeX, this is the estimate of how good the current policy is

	newTheta     [][]float64 // The policy we're currently running and thinking of switching curTheta to
	newThetaJHat float64
}

// NewTabularBBO returns an initialized TabularBBO object.
func NewTabularBBO(stateDim int, numActions int, gamma float64, N int) Agent {
	bbo := TabularBBO{}

	bbo.ep = NewEpisodeTracker(N)
	bbo.numStates = stateDim

	bbo.newTheta = make([][]float64, bbo.numStates)
	bbo.curTheta = make([][]float64, bbo.numStates)

	initialValue := 10.0
	bbo.newTheta = mathlib.Matrix(bbo.numStates, bbo.numActions, initialValue)
	bbo.curTheta = mathlib.Matrix(bbo.numStates, bbo.numActions, initialValue)

	return &bbo
}

// UpdateBeforeNextAction makes an update to the agent's policy before selecting the next action.
func (bbo *TabularBBO) UpdateBeforeNextAction() bool {
	return false
}

// EpisodicAgent returns whether the agent makes end of episode updates
func (bbo *TabularBBO) EpisodicAgent() bool {
	return true
}

// GetAction returns the action that the agent selects from the state.
func (bbo *TabularBBO) GetAction(s []float64, rng *mathlib.Random) int {
	// Convert the one-hot state into an integer from 0 - (numStates-1)
	state := mathlib.FromOneHot(s)
	// Get the action probabilities from theta, using softmax action selection.
	actionProbabilities := bbo.newTheta[state]
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
	for a := 0; a < bbo.numActions; a++ {
		sum += actionProbabilities[a]
		if temp <= sum {
			return a // The function will return 'a'. This stops the for loop and returns from the function.
		}
	}
	return bbo.numActions - 1 // Rounding error
}

// NewEpisode tells the agent that it is at the start of a new episode.
func (bbo *TabularBBO) NewEpisode() {}

// Reset the agent entirely - to a blank slate prior to learning
func (bbo *TabularBBO) Reset(rng *mathlib.Random) {
	bbo.ep.Wipe()
}

// UpdateSARS is unimplemented for this class.
func (bbo *TabularBBO) UpdateSARS(s []float64, a int, r float64, sPrime []float64, rng *mathlib.Random) {
	// Shouldn't be using this function
	panic("UpdateSARS is not implemented for TabularBBO.")
}

// UpdateSARSA - given a (s,a,r,s',a') tuple
func (bbo *TabularBBO) UpdateSARSA(s []float64, a int, r float64, sPrime []float64, aPrime int, rng *mathlib.Random) {
	bbo.ep.Update(s, a, r, s)
}

// LastUpdate lets the agent update/learn when sPrime would be the terminal absorbing state.
func (bbo *TabularBBO) LastUpdate(s []float64, a int, r float64, rng *mathlib.Random) {
	// If ready to update, update and wipe the states, actions, and rewards.
	if bbo.ep.LastUpdate(s, a, r) {
		bbo.episodeLimitReached(rng)
	}
}

func (bbo *TabularBBO) episodeLimitReached(rng *mathlib.Random) {
	bbo.episodicUpdate(rng)
	bbo.ep.Wipe()
}

// EpisodicUpdate the agent after N episodes (specified in constructor)
func (bbo *TabularBBO) episodicUpdate(rng *mathlib.Random) {
	// We are going to compute newThetaJHat (an estimate of how good the new policy is),
	// and will then see if it is better than the best policy we found so far.
	bbo.newThetaJHat = 0

	// Loop over the N episodes
	for ep := 0; ep < bbo.ep.N; ep++ {
		// Compute the return
		curGamma := 1.0
		epLen := len(bbo.ep.rewards[ep])
		for t := 0; t < epLen; t++ {
			bbo.newThetaJHat += curGamma * bbo.ep.rewards[ep][t]
			curGamma *= bbo.gamma
		}
	}
	bbo.newThetaJHat /= float64(bbo.ep.N)

	// Is the new policy better than our current best?
	if bbo.newThetaJHat > bbo.curThetaJHat {
		// It looks like it! Change our current best
		bbo.curTheta = bbo.newTheta
		bbo.curThetaJHat = bbo.newThetaJHat
	}

	// If we randomly sample policies forever, the average performance won't go up.
	// For the last 10% of episodes, we'll just run the best policy we found so far.
	for s := 0; s < bbo.numStates; s++ {
		for a := 0; a < bbo.numActions; a++ {
			bbo.newTheta[s][a] += rng.Float64()*4 - 2
		}
	}
}
