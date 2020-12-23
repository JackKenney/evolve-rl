package internal

import (
	"fmt"
	"os"
	"strconv"
	"sync"

	"github.com/jackkenney/evolve-rl/mathlib"
)

// RunEpisode calculates the return from the episode of running the agent in the environment.
func RunEpisode(
	agt Agent,
	env Environment,
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

// RunAgentEnvironment runs the passed agent in the environment passed and returns the
func RunAgentEnvironment(
	agt Agent,
	env Environment,
	numEps int,
	gamma float64,
	rng *mathlib.Random,
) []float64 {

	// Wipe the agent to start a new trial
	agt.Reset(rng)

	result := make([]float64, numEps)
	// Loop over episodes
	for epCount := 0; epCount < numEps; epCount++ {
		result[epCount] = RunEpisode(agt, env, gamma, rng)
	}

	// Return the "result" variable, holding the returns from each episode.
	return result
}

type agentConstructor func() Agent
type environmentConstructor func() Environment

// RunTrials runs them in parallel using constructors passed as arguments
func RunTrials(rng *mathlib.Random,
	agentConstructor agentConstructor,
	envConstructor environmentConstructor,
	numTrials int,
	fileName string,
) {

	// Create objects we will use
	env := envConstructor()

	// Get environment settings
	numEps := env.GetMaxEps()
	gamma := env.GetGamma()

	// Create a matrix to store the resulting returns. results(i,j) = the return on the j'th episode of the i'th trial.
	var returns = mathlib.Matrix(numTrials, numEps, 0)

	fmt.Println("Starting trial 1 of ", numTrials)

	var wg sync.WaitGroup
	// Loop over trials
	for trial := 0; trial < numTrials; trial++ {
		// % means "mod"
		if (trial+1)%1 == 0 {
			fmt.Println("Starting trial ", trial+1, " of ", numTrials)
		}
		wg.Add(1)
		go func(i int) {
			env := envConstructor()
			agt := agentConstructor()

			returns[i] = RunAgentEnvironment(agt, env, numEps, gamma, rng)

			wg.Done()
		}(trial)
	}
	wg.Wait()

	// Convert returns into a vector of mean returns and the standard error (used for error bars)
	meanReturns := mathlib.Vector(numEps, 0)
	stderrReturns := mathlib.Vector(numEps, 0)

	for epCount := 0; epCount < numEps; epCount++ {
		returns := mathlib.Column(returns, epCount)
		meanReturns[epCount] = mathlib.Mean(returns)
		stderrReturns[epCount] = mathlib.StdError(returns)
	}

	// Print the results to a file
	file, err := os.Create(fileName + "_out.csv")
	if err != nil {
		fmt.Println(err.Error() + "\n")
	}
	defer file.Close()
	file.WriteString("REINFORCE,BBO,REINFORCE Error Bar,BBO Error Bar\n")
	var line string
	for epCount := 0; epCount < numEps; epCount++ {
		line = strconv.FormatFloat(meanReturns[epCount], 'g', -1, 64) + "," + strconv.FormatFloat(stderrReturns[epCount], 'g', -1, 64)
		file.WriteString(line + "\n")
	}
}
