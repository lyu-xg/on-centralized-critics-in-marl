
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# lib for Multi-agent Reinforcement Learning

## Setup

Make sure you have your [mongodb](https://docs.mongodb.com/manual/administration/install-community/) running locally.

## Running experiments

```bash
python main.py with ppo
python main.py with ac
```

```bash
python main.py with n_agents=2
python main.py with ppo seed=42 discount_factor=0.995
```

## Visualizing runs

install [omniboard](https://www.npmjs.com/package/omniboard) and run

```bash
omniboard -m localhost:27017:experiments
```

after which you should be able to connect to [localhost:9000](http://localhost:9000) to access the dashboard.

## acknowledgement

Some code was taken from [this repo](https://github.com/adik993/ppo-pytorch)