# Jack Kenney 2020
GO       := go
GO_FLAGS := build

EXECUTABLE := main
WIRE := wire
BIN     := bin
CMD  := cmd
INTERNAL := internal

all: $(CMD)/*.go
	$(GO) $(GO_FLAGS) -o $(BIN)/$(EXECUTABLE) $^

clean: 
	find bin/* -type f -not -name 'BIN.md' -delete
