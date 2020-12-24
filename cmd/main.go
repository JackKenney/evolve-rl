package main

import (
	"fmt"
	"os/exec"
	"sync"

	"github.com/jackkenney/evolve-rl/internal"
	"github.com/jackkenney/evolve-rl/mathlib"
)

func runner(fileName string) {
	numTrials := 1

	// Hyperparameters
	bboN := 10
	alphaSarsa := 0.001
	alphaReinforce := 0.001
	optimisticValue := 10.0

	rng := mathlib.NewRandom(0)
	env := internal.NewGridworld(rng)

	// Get some values once so we don't have to keep looking them up (i.e., so we can type less later)
	stateDim := env.GetStateDim()
	numActions := env.GetNumActions()
	gamma := env.GetGamma()

	// Constructors used by parallelized trials
	envConstructor := func() internal.Environment {
		return internal.NewGridworld(rng)
	}
	agtConstructor := func() internal.Agent {
		if fileName == "sarsa" {
			return internal.NewSarsa(stateDim, numActions, gamma, alphaSarsa, optimisticValue)
			// } else if fileName == "q-learning" {
			// 	return internal.NewTabularBBO(stateDim, numActions, gamma, bboN)
		} else if fileName == "reinforce" {
			return internal.NewREINFORCE(stateDim, numActions, gamma, alphaReinforce)
		} else if fileName == "bbo" {
			return internal.NewTabularBBO(stateDim, numActions, gamma, bboN)
		} else {
			panic("No algorithm selected")
		}
	}

	// Run parallel trials
	internal.RunTrials(rng, agtConstructor, envConstructor, numTrials, fileName)
}

func main() {
	algorithms := []string{"sarsa", "reinforce", "bbo"} //, "q-learning"}

	// Run algorithms in parallel
	var wg sync.WaitGroup
	for _, fileName := range algorithms {
		wg.Add(1)
		go func(f string) {
			runner(f)
			wg.Done()
		}(fileName)
	}
	wg.Wait()

	// Make plots
	for _, fileName := range algorithms {
		wg.Add(1)
		go func(f string) {
			cmd := exec.Command("matlab", "-batch", "'plotResults(\""+f+"_out"+"\")'")
			err := cmd.Run()

			if err != nil {
				fmt.Printf("%s\n", err)
			}

			wg.Done()
		}(fileName)
	}
	wg.Wait()
}
