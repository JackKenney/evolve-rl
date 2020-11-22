package internal

import (
	"github.com/google/wire"
)

// DefaultProviderSet injects the dependencies from the Model through to the Router.
var DefaultProviderSet = wire.NewSet(
	wire.Bind(new(environment.Environment), new(*environment.GridWorld)),
)
