// +build wireinject
// The build tag makes sure the stub is not built in the final build.

package main

import (
	"github.com/evolve-rl/internal"
	"github.com/google/wire"
)

// "github.com/google/wire"

func InjectDriver() (*internal.Driver, error) {
	wire.Build(internal.DefaultProviderSet)
	return &internal.Driver{}, nil
}
