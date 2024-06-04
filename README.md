# Neurality

Playing around with neurons

- [Neurality](#neurality)
  - [Setup](#setup)
  - [Architecture](#architecture)
    - [NeuronConnection](#neuronconnection)
    - [Neuron](#neuron)
  - [How it works](#how-it-works)
  - [Evolution simulation](#evolution-simulation)
    - [Variables](#variables)
  - [TODO](#todo)

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

## Evolution simulation

Creatures will spawn in a room with 5 doors. A creature can choose from 3 grey doors, 1 green door and 1 red door:

- Grey door leads to nowhere; Creature stays in the same room.
- Green door leads to the next room; Creature advances to the next room.
- Red door leads to a trap; Creature dies.

Creature has to choose the green door to proceed to the next room, and so on. The creature that advances to at least room number 10 will survive.

Simulation will take 200 cycles. Each creature will have 3-10 cycles to "think" in each room before choosing a door.

### Variables

Inputs | Outputs
-|-
Door 1 type | Prob. of choosing Door 1
Door 2 type | Prob. of choosing Door 2
Door 3 type | Prob. of choosing Door 3
Door 4 type | Prob. of choosing Door 4
Door 5 type | Prob. of choosing Door 5

Door type will be defined as follows:

Door type | Value
-|-
Red | `-1.0`
Gray | `0.0`
Green | `1.0`

## TODO

- Optimize performance of creating networks
- Optimize performance of computing
