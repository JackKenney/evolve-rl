package main

import (
	"fmt"
	"math/rand"
	"os"
	"strconv"

	"github.com/jackkenney/evolve-rl/internal"
	"github.com/jackkenney/evolve-rl/mathlib"
)

func runAgentEnvironment(
	agt internal.Agent, // The agent to run. The * here means that this is a pointer to an agent object. "pointer" means "memory location". So, this function takes as input the location of an object in memory, and that object satisfies the spceifications of the "Agent" class.
	env internal.Environment, // The environment to run on (pointer)
	maxEps int, // The number of episodes to run
	gamma float64, // The discount factor to use
	rng *rand.Rand) []float64 { // Random number rng to use.

	// Wipe the agent to start a new trial
	agt.Reset(rng)

	// Create variables that we will use
	var curAction, newAction int
	var reward, curGamma float64
	var result, curState, newState []float64

	// Loop over episodes
	for epCount := 0; epCount < maxEps; epCount++ {
		// Tell the agent and environment that we're starting a new episode. For the first episode, this may be redundant if the agent and environment were just created
		env.NewEpisode(rng)
		agt.NewEpisode()

		// Prepare for the new episode
		result = append(result, 0)               // We will store the return here
		curGamma = 1                             // We will store gamma^t here
		curState = env.GetState()                // Get the initial state
		curAction = agt.GetAction(curState, rng) // Get the initial action

		// Loop over time
		for t := 0; true; t++ {
			if t <= 100 {
				fmt.Printf(strconv.FormatInt(int64(t), 10) + " ")
			}
			reward = env.Transition(curAction, rng) // Update the state of the environment and get the reward
			result[epCount] += curGamma * reward    // Update the return for this episode
			curGamma *= gamma                       // Decay curGamma

			if env.InTAS() { // Check if in the terminal absorbing state
				agt.LastUpdate(curState, curAction, reward, rng) // In the terminal absorbing state, so do a special temrinal update
				break                                            // Break out of the loop over time.
			}

			newState = env.GetState()         // If we get here, the new state isn't the terminal absorbing state. Get the new state.
			if agt.UpdateBeforeNextAction() { // Check if we should update before computing the next action
				agt.UpdateSARS(curState, curAction, reward, newState, rng) // Update before getting the new action
				newAction = agt.GetAction(newState, rng)                   // Get the new action
			} else {
				newAction = agt.GetAction(newState, rng)                               // Get the new action before updating the agent
				agt.UpdateSARSA(curState, curAction, reward, newState, newAction, rng) // Update the agent
			}

			// Prepare for the next iteration of the t-loop, where "new" variables will be the "cur" variables.
			curAction = newAction
			curState = newState
		}
	}

	// Return the "result" variable, holding the returns from each episode.
	return result
}

func main() {
	// Create objects we will use
	source := rand.NewSource(0) // Random number rng, seeded with zero
	rng := rand.New(source)
	env := internal.NewGridworld(rng) // Create the environment, in this case a Gridworld
	numTrials := 100                  // Specify the number of trials to run and get the number of episodes per tiral.

	// Get some values once so we don't have to keep looking them up (i.e., so we can type less later)
	stateDim := env.GetStateDim()
	numActions := env.GetNumActions()
	maxEps := env.GetMaxEps()
	gamma := env.GetGamma()

	/////
	// If you wnat to use the manual agent, uncomment the line below, and comment out the line defining the agent to be a TabularRandomSearch object.
	/////
	//Manual a1;					// Create the agent
	N := 10 // How many episodes are run between update calls within TabularRandomSearch?

	// a1 := internal.REINFORCE(stateDim, numActions, gamma) //  REINFORCE doesn't take N, since it will always use N = 1.
	a2 := internal.NewTabularBBO(stateDim, numActions, gamma, N, maxEps)

	// returnsA1 := mathlib.Matrix(numTrials, maxEps, 0) // Create a matrix to store the resulting returns. results(i,j) = the return on the j'th episode of the i'th trial.
	returnsA2 := mathlib.Matrix(numTrials, maxEps, 0) // Same as above, but for the second agent
	fmt.Println("Starting trial 1 of ", numTrials+1)  // "cout" means "console out", and is our print command. Separate objects to print with the << symbol. Here we are printing a string, followed by an integer, followed by std::endl (end line).
	for trial := 0; trial < numTrials; trial++ {      // Loop over trials
		if (trial+1)%1 == 0 { // % means "mod"
			fmt.Println("Starting trial ", trial+1, " of ", numTrials)
		}
		// Run the agent on the environment for this trial, and store the result in the trial'th row of returns.
		// The & before a1 and env indicates that we are passing pointers to a1, a2, and env. This is because runAgentEnvironment
		// won't know the type of the agent and environment, only that they meet the specifications of Agent.hpp and Environment.hpp.
		// So, on their end, these inputs are pointers to objects of unknown exact type, but which meet the Agent/Environment specifications.
		// returnsA1[trial] = runAgentEnvironment(&a1, &env, maxEps, gamma, a1.updateBeforeNextAction(), rng).transpose()
		returnsA2[trial] = runAgentEnvironment(a2, env, maxEps, gamma, rng)
	}

	// Convert returns into a vector of mean returns and the standard error (used for error bars)
	// meanReturnsA1 := mathlib.Vector(maxEps)
	// stderrReturnsA1 := mathlib.Vector(maxEps)
	meanReturnsA2 := mathlib.Vector(maxEps, 0)
	stderrReturnsA2 := mathlib.Vector(maxEps, 0)

	for epCount := 0; epCount < maxEps; epCount++ {
		// meanReturnsA1[epCount] = returnsA1.col(epCount).mean()
		// stderrReturnsA1[epCount] = mathlib.StdError(returnsA1.col(epCount))
		returns := mathlib.Column(returnsA2, epCount)
		meanReturnsA2[epCount] = mathlib.Mean(returns)
		stderrReturnsA2[epCount] = mathlib.StdError(returns)
	}

	// Print the results to a file
	file, err := os.Create("out.csv") // Create an "output file stream". This will actually create the file, but it will be empty
	if err != nil {
		fmt.Println(err.Error() + "\n")
	}
	defer file.Close()
	// file.WriteString("REINFORCE,BBO,TRS Error Bar,BBO Error Bar\n")
	file.WriteString("BBO,BBO Error Bar\n")
	var line string
	for epCount := 0; epCount < maxEps; epCount++ {
		// line = strconv.FormatFloat(meanReturnsA1[epCount], 'g', -1, 64) + "," + strconv.FormatFloat(meanReturnsA2[epCount], 'g', -1, 64) + ',' + strconv.FormatFloat(stderrReturnsA1[epCount], 'g', -1, 64) + "," + strconv.FormatFloat(stderrReturnsA2[epCount], 'g', -1, 64)
		line = strconv.FormatFloat(meanReturnsA2[epCount], 'g', -1, 64) + "," + strconv.FormatFloat(stderrReturnsA2[epCount], 'g', -1, 64)
		file.WriteString(line + "\n")
	}
}
