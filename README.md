<h1 align='center'>Adaptive SDE solvers for the SABR model</h1>
<h2 align='center'>by James Foster and Andraž Jelinčič</h2>

This repository contains numerical examples for the paper
@misc{foster2024convergence,
      title={On the convergence of adaptive approximations for stochastic differential equations}, 
      author={James Foster and Andraž Jelinčič},
      year={2024},
      eprint={2311.14201},
      archivePrefix={arXiv},
      primaryClass={math.NA},
      url={https://arxiv.org/abs/2311.14201}, 
}

This code is based on the Diffrax package by Patrick Kidger, which is available at [github.com/patrick-kidger/diffrax](https://github.com/patrick-kidger/diffrax).
The documentation for Diffrax can be found at [docs.kidger.site/diffrax](https://docs.kidger.site/diffrax).

The code for the SABR model example is in `notebooks/SABR.ipynb`.

To use this code, clone the repository and install the requirements:

```bash
git clone https://github.com/andyElking/Adaptive_SABR.git
cd Adaptive_SABR/
python -m venv venv
source venv/bin/activate
pip install -e .
```
If you aren't using a virtual environment, you can omit the third and fourth lines.
