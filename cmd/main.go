package main

import (
	"fmt"
	"os"
	"strconv"
	"sync"

	"github.com/jackkenney/evolve-rl/internal"
	"github.com/jackkenney/evolve-rl/mathlib"
)

func runEpisode(
	agt internal.Agent,
	env internal.Environment,
	gamma float64,
	rng *mathlib.Random,
) float64 {
	// Prepare objects
	env.NewEpisode(rng)
	agt.NewEpisode()

	// Create variables that we will use
	var curAction, newAction int
	var result, reward, curGamma float64
	var curState, newState []float64

	// Prepare for the new episode
	result = 0
	curGamma = 1
	curState = env.GetState()
	curAction = agt.GetAction(curState, rng)

	// Loop over time
	for t := 0; true; t++ {
		reward = env.Transition(curAction, rng)
		result += curGamma * reward
		curGamma *= gamma

		// Check if in the terminal absorbing state
		if env.InTAS() {
			agt.LastUpdate(curState, curAction, reward, rng)
			break
		}

		newState = env.GetState()

		// Check if we should update before computing the next action
		if agt.UpdateBeforeNextAction() {
			agt.UpdateSARS(curState, curAction, reward, newState, rng)
			newAction = agt.GetAction(newState, rng)
		} else {
			newAction = agt.GetAction(newState, rng)
			agt.UpdateSARSA(curState, curAction, reward, newState, newAction, rng)
		}

		// Prepare for the next iteration of the t-loop, where "new" variables will be the "cur" variables.
		curAction = newAction
		curState = newState
	}
	return result
}
func runAgentEnvironment(
	agt internal.Agent,
	env internal.Environment,
	maxEps int,
	gamma float64,
	rng *mathlib.Random,
) []float64 {

	// Wipe the agent to start a new trial
	agt.Reset(rng)

	result := make([]float64, maxEps)
	// Loop over episodes
	for epCount := 0; epCount < maxEps; epCount++ {
		result[epCount] = runEpisode(agt, env, gamma, rng)
	}

	// Return the "result" variable, holding the returns from each episode.
	return result
}

func main() {
	// Create objects we will use
	rng := mathlib.NewRandom(0)
	numTrials := 1000
	env := internal.NewGridworld(rng)

	// Get some values once so we don't have to keep looking them up (i.e., so we can type less later)
	stateDim := env.GetStateDim()
	numActions := env.GetNumActions()
	maxEps := env.GetMaxEps()
	gamma := env.GetGamma()

	// How many episodes are run between update calls within TabularRandomSearch?
	N := 10

	// Create a matrix to store the resulting returns. results(i,j) = the return on the j'th episode of the i'th trial.
	var returnsA1 = mathlib.Matrix(numTrials, maxEps, 0)
	var returnsA2 = mathlib.Matrix(numTrials, maxEps, 0)

	fmt.Println("Starting trial 1 of ", numTrials+1)

	var wg sync.WaitGroup
	// Loop over trials
	for trial := 0; trial < numTrials; trial++ {
		// % means "mod"
		if (trial+1)%1 == 0 {
			fmt.Println("Starting trial ", trial+1, " of ", numTrials)
		}
		wg.Add(1)
		go func(i int) {
			// create objects
			env := internal.NewGridworld(rng)
			a1 := internal.NewREINFORCE(stateDim, numActions, gamma)
			a2 := internal.NewTabularBBO(stateDim, numActions, gamma, N)
			// get returns from history
			returnsA1[i] = runAgentEnvironment(a1, env, maxEps, gamma, rng)
			returnsA2[i] = runAgentEnvironment(a2, env, maxEps, gamma, rng)

			wg.Done()
		}(trial)
	}
	wg.Wait()

	// Convert returns into a vector of mean returns and the standard error (used for error bars)
	meanReturnsA1 := mathlib.Vector(maxEps, 0)
	stderrReturnsA1 := mathlib.Vector(maxEps, 0)
	meanReturnsA2 := mathlib.Vector(maxEps, 0)
	stderrReturnsA2 := mathlib.Vector(maxEps, 0)

	for epCount := 0; epCount < maxEps; epCount++ {
		returns := mathlib.Column(returnsA1, epCount)
		meanReturnsA1[epCount] = mathlib.Mean(returns)
		stderrReturnsA1[epCount] = mathlib.StdError(returns)

		returns = mathlib.Column(returnsA2, epCount)
		meanReturnsA2[epCount] = mathlib.Mean(returns)
		stderrReturnsA2[epCount] = mathlib.StdError(returns)
	}

	// Print the results to a file
	file, err := os.Create("out.csv")
	if err != nil {
		fmt.Println(err.Error() + "\n")
	}
	defer file.Close()
	file.WriteString("REINFORCE,BBO,REINFORCE Error Bar,BBO Error Bar\n")
	var line string
	for epCount := 0; epCount < maxEps; epCount++ {
		line = strconv.FormatFloat(meanReturnsA1[epCount], 'g', -1, 64) + "," + strconv.FormatFloat(meanReturnsA2[epCount], 'g', -1, 64) + "," + strconv.FormatFloat(stderrReturnsA1[epCount], 'g', -1, 64) + "," + strconv.FormatFloat(stderrReturnsA2[epCount], 'g', -1, 64)
		file.WriteString(line + "\n")
	}
}
