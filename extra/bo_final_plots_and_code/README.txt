Final BO figure package
=======================
Contains only the final nine acquisition-color variants, each with white and
transparent backgrounds. All use the final blue posterior, black axes, 9 pt
text, 219 x 200 TeX pt canvas, 2:1 panel heights, and adjusted dashed guides.

The plotting script is standalone and uses plots/bo_snapshot.npz. No TuneControl
checkout or new simulations are needed to reproduce these figures.

From the extracted package root, use Python 3.12 and install dependencies:
  python3.12 -m venv .venv
  .venv/bin/python -m pip install -r plots/requirements-local.txt

The script also renders comparison images, so Poppler's pdftocairo must be
available on PATH (for example, install Poppler using your package manager).
Run:
  .venv/bin/python plots/bo_acquisition_colors.py

The data file contains the original observations and the sampled posterior
mean, standard deviation, acquisition function, and selected next point.
See plots/acquisition_colors/README.md for PDF links and color names.
