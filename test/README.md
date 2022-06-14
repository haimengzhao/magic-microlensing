## Experiments

This folder contains the Jupyter Notebooks for tests and experiements showed in the paper.

### Locator
Tests of locator can be found in [`test_locator.ipynb`](./test_locator.ipynb)

### Estimator
Tests of estimator can be found in [`test_cdemdn.ipynb`](./test_cdemdn.ipynb) (the estimator without optimization), [`test_opt.ipynb`](./test_opt.ipynb) (the optimization step) and [`test_lcplot.ipynb`](./test_lcplot.ipynb) (plots of light curve examples).

### Application
Tests of the joint pipeline are in [`loc+cdemdn.ipynb`](./loc+cdemdn.ipynb).

An example of applying MAGIC to a real event ([KMT-2019-BLG-0371](https://iopscience.iop.org/article/10.3847/1538-3881/abf930)) is given in [`test_KMT.ipynb`](./test_KMT.ipynb).

[The `./KMT/` folder](./KMT/) contains some attempts of applying MAGIC to more real events.

### Extended Abstract
Experiments in the extended abstract can be found in [`analysis.ipynb`](./analysis.ipynb) and [`plot.ipynb`](./plot.ipynb).

### Miscellanies
[`rescale.ipynb`](./rescale.ipynb) explores the differences between the impact of different microlensing parameters. 

[`opt.ipynb`](./opt.ipynb) and [`downhill_optimization.ipynb`](./downhill_optimize.ipynb) explores how to automate the optimization step for batches of light curves.

[`locate_and_scale.py`](./locate_and_scale.py) is a automated script for transforming (i.e. shifting and rescaling) light curves given $t_0$ and $t_E$.

[`plot_triangle.py`](./plot_triangle.py) is a package for drawing contours of a Gaussian mixture.

Note that the python scripts (ending with `.py`) are normally the massive production version of the coressponding Jupyter notebooks.
