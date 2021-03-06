package internal

import "github.com/jackkenney/evolve-rl/mathlib"

// Gridworld object which extends the Environment interface
type Gridworld struct {
	x   int  // Agent horizontal coordinate (0 to 4)
	y   int  // Agent vertical coordinate (0 to 4)
	t   int  // Time into the episode. We terminate after 100 time steps
	tas bool // Are we in the terminal absorbing state?
}

// NewGridworld returns a new Gridworld Environment object.
func NewGridworld(rng *mathlib.Random) Environment {
	env := Gridworld{}
	env.NewEpisode(rng)
	return &env
}

// GetMaxEps returns how many episodes should be run
func (env *Gridworld) GetMaxEps() int {
	return 1000
}

// GetStateDim returns the dimension (length) of state vectors.
func (env *Gridworld) GetStateDim() int {
	return 23
}

// GetNumActions returns |\mathcal A|. Note that we are assuming that the action set is finite.
func (env *Gridworld) GetNumActions() int {
	return 4
}

// GetGamma returns \gamma
func (env *Gridworld) GetGamma() float64 {
	return 0.9
}

// Transition applies action a, updating the state of the environment. It returns the reward that results from the state transition.
func (env *Gridworld) Transition(a int, r *mathlib.Random) float64 {
	// Count the next timestep
	env.t++

	// Check if we should transition to s_infty.
	if env.x == 4 && env.y == 4 {
		env.tas = true
		return 0.0
	}

	// Check if we should terminate due to running too long
	if env.t == 100 {
		env.tas = true
		return -100.0
	}

	// We implement the "veer" and "stay" behavior with an "effective action" that is modified from the actual action "a"
	effectiveAction := a

	// Temp is a uniform random number from 0 to one.
	temp := r.Float64()

	if temp <= 0.1 {
		// This should pass 10% of the time
		effectiveAction = -1 // Actions -1 causes the "stay" behavior below
	} else if temp <= 0.15 {
		// This should pass (but not the previous if-statement) 5% of the time.
		effectiveAction++ // Rotate the action 90 degrees
		if effectiveAction == 4 {
			effectiveAction = 0 // Wrap around
		}
	} else if temp <= 0.2 {
		// This should happen 5% of the time
		effectiveAction-- // Rotate the action -90 degrees
		if effectiveAction == -1 {
			effectiveAction = 3 // Wrap around
		}
	}

	// Compute the resulting agent position
	xPrime := env.x
	yPrime := env.y
	if (effectiveAction == 0) && (env.y >= 1) {
		yPrime-- // Up
	} else if effectiveAction == 1 {
		xPrime++ // Right
	} else if effectiveAction == 2 {
		yPrime++ // Down
	} else if effectiveAction == 3 {
		xPrime-- // Left
	}

	// If the new position is valid, then update to it!
	if (xPrime >= 0) && (yPrime >= 0) && (xPrime < 5) && (yPrime < 5) &&
		((xPrime != 2) || ((yPrime != 2) && (yPrime != 3))) {
		env.x = xPrime
		env.y = yPrime
	}

	// Compute the resulting reward
	if (env.x == 2) && (env.y == 4) {
		// The agent is in the "water" state
		return -10
	} else if (env.x == 4) && (env.y == 4) {
		// The agent is in the bottom-right "goal" state.
		return 10
	} else {
		// The agent isn't in the "water" or the "goal" state, so the reward is zero
		return 0
	}
}

// GetState returns the current state of the environment
func (env *Gridworld) GetState() []float64 {
	// Check if we should transition to s_infty.
	if env.x == 4 && env.y == 4 {
		env.tas = true
	}

	// Create the object we will return, and initialize to the zero-vector, of length 23.
	result := make([]float64, 23)
	if !env.tas {
		state := env.y*5 + env.x

		// Do not enter obstacles
		if state == 12 {
			panic("Agent inside of upper obstacle")
		}
		if state > 12 {
			state--
		}
		if state == 16 {
			panic("Agent inside of lower obstacle")
		}
		if state > 16 {
			state--
		}
		result[state] = 1.0
	} else {
		panic("GetState called when in TAS.")
	}
	// Return the computed state representation
	return result
}

// InTAS returns whether the current state is Terminal Absorbing State (TAS)
func (env *Gridworld) InTAS() bool {
	return env.tas || (env.x == 4 && env.y == 4)
}

// NewEpisode resets the environment to start a new episode (it samples the state from the initial state distribution).
func (env *Gridworld) NewEpisode(rng *mathlib.Random) {
	// Start at position (0,0), with the time counter also equal to zero
	env.x = 0
	env.y = 0
	env.t = 0
	// We do not start in the terminal absorbing state
	env.tas = false
}
