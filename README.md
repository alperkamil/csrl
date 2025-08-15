# CSRL v2: Control Synthesis from Formal Specifications using Reinforcement Learning
CSRL v2 provides implementations for synthesizing control policies from formal task specifications using reinforcement learning.

## ✨ Features
### Formal Task Specifications
- Linear Temporal Logic (**LTL**)
- Limit-Deterministic Büchi Automata (**LDBA**)
- Deterministic Parity Automata (**DPA**)
- Reward Machines (**RM**)

### Environments
- Markov Decision Processes (**MDPs**)
  - GridWorld (discrete)
- Stochastic Games (**SGs**)
  - Zero-sum, turn-based, two-player GridWorld with an actuation attacker
- Product Enviroments
  - Product MDPs and SGs with automata/specs

### Control Synthesis
- **Value Iteration** for MDPs
- **Shapley Algorithm** for SGs (coming soon)
- **Q-learning** for MDPs (coming soon)
- **Minimax-Q** for SGs (coming soon)


## 🚀 Installation
Please check [INSTALL.md](./INSTALL.md) for installation details.

### Requirements

- [**Python**](https://www.python.org/) ≥ 3.10
- [**Owl**](https://owl.model.in.tum.de/) ≥ 21.0 — `ltl2ldba` and `ltl2dpa` binaries must be in your `PATH`
- [**Spot**](https://spot.lrde.epita.fr/) ≥ 2.11 — with Python bindings installed in your environment

### Basic Installation
```bash
git clone https://github.com/alperkamil/csrl
cd csrl
pip install -e .
```


## 📖 Citation
If this repository supports your research, we'd appreciate it if you could cite the relevant work listed in [CITATION.bib](./CITATION.bib).


## 💐 Acknowledgments
This project builds on excellent tooling from:
- [**Owl**](https://owl.model.in.tum.de/) for LTL-to-automata compilation
- [**Spot**](https://spot.lrde.epita.fr/) for temporal logic and automata manipulation


## 🛠️ Troubleshooting
If you run into any problems, please feel free to open an issue and include as many details as possible.