# Neurality

Playing around with neurons

## Architecture

### NeuronConnection

- Weight
- Value

### Neuron

- Bias
- Value

## How it works

Singular loop:

  - Neurons gather values from connections
  - Neurons calculate their value
  - Connections value reset to 0
  - Neuron add the value to the active connection value.

## TODO

Lay out them into different numpy arrays:
  - W (weights)
  - B (biases)
  - I (neuron input values)
  - O (neuron output values)

And then compute.