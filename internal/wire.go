package internal

import (
	"github.com/google/wire"
	environment "github.com/jackkenney/evolve-rl/internal/environment"
)

// DefaultProviderSet injects the dependencies from the Model through to the Router.
var DefaultProviderSet = wire.NewSet(
	wire.Bind(new(environment.Environment), new(*environment.GridWorld)),
)
