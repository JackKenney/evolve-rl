# Evolve Reinforcement Learning

A parallelized reinforcement learning library written in Golang that 
supports parallelized populations of learners where parameters and hyperparameters evolve over generations and parameters are learned intra-generationally through "epigenetic" model.

## Prerequisites

* Golang 1.13.8
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

