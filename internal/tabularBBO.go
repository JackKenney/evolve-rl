package internal

import (
	"math"
	"math/rand"
)

// TabularBBO learning agent using black box optimization
type TabularBBO struct {
	N       int           // How many episodes before update?
	states  [][][]float64 // States history
	actions [][]int       // Action history
	rewards [][]float64   // Rewards history

	numStates  int     // How many discrete states?
	numActions int     // How many discrete actions?
	maxEps     int     // How many episodes will be run?
	epCount    int     // Track how many episodes have been run.
	gamma      float64 // Discount parameter

	curTheta     [][]float64 // The current best policy we have found
	curThetaJHat float64     // $\hat J(\theta_\text{cur})$ in LaTeX, this is the estimate of how good the current policy is
	newTheta     [][]float64 // The policy we're currently running and thinking of switching curTheta to
	newThetaJHat float64
}

// NewTabularBBO returns an initialized TabularBBO object.
func NewTabularBBO(stateDim int, numActions int, gamma float64, N int, maxEps int) EpisodicAgent {
	agt := TabularBBO{}
	agt.numStates = stateDim
	agt.newTheta = make([][]float64, agt.numStates)
	agt.curTheta = make([][]float64, agt.numStates)

	initialValue := 10.0
	for i := 0; i < agt.numStates; i++ {
		for j := 0; j < agt.numActions; j++ {
			agt.newTheta[i][j] = initialValue
			agt.curTheta[i][j] = initialValue
		}
	}
	return agt
}

// UpdateBeforeNextAction makes an update to the agent's policy before selecting the next action.
func (bbo TabularBBO) UpdateBeforeNextAction() bool {
	return true
}

// GetAction returns the action that the agent selects from the state.
func (bbo TabularBBO) GetAction(s []float64, rng *rand.Rand) int {
	// Convert the one-hot state into an integer from 0 - (numStates-1)
	state := 0
	for state = 0; state < len(s); state++ {
		if s[state] != 0 {
			break
		}
	}
	if state == len(s) { // If this happens, the s-vector was all zeros
		panic("state vector was all zeros")
	}
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
func (bbo TabularBBO) NewEpisode() {}

// Reset the agent entirely - to a blank slate prior to learning
func (bbo TabularBBO) Reset(rng *rand.Rand) {
	bbo.epCount = 0
	bbo.wipeStatesActionsRewards()
}

// UpdateSARS is unimplemented for this class.
func (bbo TabularBBO) UpdateSARS(s []float64, a int, r float64, sPrime []float64, rng *rand.Rand) {
	if bbo.UpdateBeforeNextAction() {
		panic("UpdateSARS is not implemented for TabularBBO.")
	}
}

// UpdateSARSA - given a (s,a,r,s',a') tuple
func (bbo TabularBBO) UpdateSARSA(s []float64, a int, r float64, sPrime []float64, aPrime int, rng *rand.Rand) {
	bbo.states[bbo.epCount] = append(bbo.states[bbo.epCount], s)
	bbo.actions[bbo.epCount] = append(bbo.actions[bbo.epCount], a)
	bbo.rewards[bbo.epCount] = append(bbo.rewards[bbo.epCount], r)
}

// LastUpdate lets the agent update/learn when sPrime would be the terminal absorbing state.
func (bbo TabularBBO) LastUpdate(s []float64, a int, r float64, rng *rand.Rand) {
	bbo.states[bbo.epCount] = append(bbo.states[bbo.epCount], s)
	bbo.actions[bbo.epCount] = append(bbo.actions[bbo.epCount], a)
	bbo.rewards[bbo.epCount] = append(bbo.rewards[bbo.epCount], r)

	// Increment episode counter
	bbo.epCount++

	if bbo.epCount == bbo.N { // If ready to update, update and wipe the states, actions, and rewards
		bbo.EpisodicUpdate(rng)
		bbo.wipeStatesActionsRewards()
		bbo.epCount = 0
	}
}

// EpisodicUpdate the agent after N episodes (specified in constructor)
func (bbo TabularBBO) EpisodicUpdate(rng *rand.Rand) {
	// Track how many episodes have passed
	bbo.epCount += bbo.N

	// We are going to compute newThetaJHat (an estimate of how good the new policy is), and will then
	// see if it is better than the best policy we found so far.
	bbo.newThetaJHat = 0

	// Loop over the N episodes
	for ep := 0; ep < bbo.N; ep++ {
		// Compute the return
		curGamma := 1.0
		epLen := len(bbo.rewards[ep])
		for t := 0; t < epLen; t++ {
			bbo.newThetaJHat += curGamma * bbo.rewards[ep][t]
			curGamma *= bbo.gamma
		}
	}
	bbo.newThetaJHat /= float64(bbo.N)

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
			bbo.newTheta[s][a] = bbo.curTheta[s][a] + rng.Float64()*4 - 2
		}
	}
}

// Clear agent's memory
func (bbo TabularBBO) wipeStatesActionsRewards() {
	for i := 0; i < bbo.N; i++ {
		bbo.states[i] = [][]float64{}
		bbo.actions[i] = []int{}
		bbo.rewards[i] = []float64{}
	}
}
