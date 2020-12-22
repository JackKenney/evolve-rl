package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"

	"github.com/jackkenney/evolve-rl/internal"
)

func mean(v []float64) float64 {
	total := 0.0
	for _, number := range v {
		total = total + number
	}
	average := total / float64(len(v)) // len  function return array size
	return average
}

func stdError(v []float64) float64 {
	// The most common object types are int (integer), double (double precision floating point), bool (Boolean), VectorXd (vector), MatrixXd (matrix), and vector<type>. We'll talk about vector<type> later.
	sampleMean := mean(v) // First, get the mean of the vector
	temp := 0.0           // Create a floating point (double precision) equal to zero
	// Below the (int) term means "cast the next thing into an "int" type. v.size() actually returns a long integer. C++ will automatically cast it to an int to compare to i, but your compiler might give you a warning that you're comparing two different integer types. The explicit casting to an "int" here avoids that warning.
	for _, val := range v { // This is a basic for loop. The variable i is initialized to zero at the start, it runs as long as i < (int)v.size(), and at the end of every iteration of the loop it calls i++ (i = i + 1).
		temp += (val - sampleMean) * (val - sampleMean) // temp += foo; means the same thing as temp = temp + foo;
	}
	return math.Sqrt(temp/float64(len(v)-1.0)) / math.Sqrt(float64(len(v))) // Return the standard error. The returned object must match the return type in the function delaration.
}

func runAgentEnvironment(
	agt internal.Agent, // The agent to run. The * here means that this is a pointer to an agent object. "pointer" means "memory location". So, this function takes as input the location of an object in memory, and that object satisfies the spceifications of the "Agent" class.
	env internal.Environment, // The environment to run on (pointer)
	maxEps int, // The number of episodes to run
	gamma float64, // The discount factor to use
	updateBeforeNextAction bool, // Does this agent update before or after the next action, A_{t+1} is chosen?
	rng *rand.Rand) []float64 { // Random number generator to use.

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
		result[epCount] = 0                      // We will store the return here
		curGamma = 1                             // We will store gamma^t here
		curState = env.GetState()                // Get the initial state
		curAction = agt.GetAction(curState, rng) // Get the initial action

		// Loop over time
		for t := 0; true; t++ {
			reward = env.Transition(curAction, rng) // Update the state of the environment and get the reward
			result[epCount] += curGamma * reward    // Update the return for this episode
			curGamma *= gamma                       // Decay curGamma
			if env.InTAS() {                        // Check if in the terminal absorbing state

				agt.LastUpdate(curState, curAction, reward, rng) // In the terminal absorbing state, so do a special temrinal update
				break                                            // Break out of the loop over time.
			}
			newState = env.GetState()   // If we get here, the new state isn't the terminal absorbing state. Get the new state.
			if updateBeforeNextAction { // Check if we should update before computing the next action
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
	flag.Parse()

	fmt.Println("Hello, world.")
}
