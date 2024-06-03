# Neurality

Playing around with neurons

## Setup

Install dependencies:

```bash
sudo apt install python3.10 graphviz
```

Create python virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3.10 main.py
```

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