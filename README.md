# Evolve Reinforcement Learning

A parallelized reinforcement learning library written in Golang that 
supports parallelized populations of learners where parameters and hyperparameters evolve over generations and parameters are learned intra-generationally through "epigenetic" model.

## Prerequisites

* Golang (version in `go.mod` file)
* GNU Make 4.2.1

Install modules
```bash
go mod download
```

### Verify projects makes successfully
```
make
```

### Run the tests

```bash
make tests
```

## Acknowledgements

Translated and modified from Phil Thomas's COMPSCI 687 at UMass CICS.
