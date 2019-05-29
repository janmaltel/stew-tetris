# Regularization in directable environments with application to Tetris

This repository contains a Python implementation of M-learning with shrinkage toward equal weights (STEW) regularization applied to Tetris, as used in the article:

Lichtenberg, J. M. & Şimşek, Ö. (2019). [Regularization in directable environments with application to Tetris](http://proceedings.mlr.press/v97/lichtenberg19a.html). *Proceedings of the 36th International Conference on Machine Learning, in PMLR* 97:3953-3962

Further implementation details and pseudo-code of M-learning are available in the [Supplementary Material](http://proceedings.mlr.press/v97/lichtenberg19a/lichtenberg19a-supp.pdf).

## Installation
Install required Python packages via

`pip install -r requirements.txt`

## Run
The following command runs M-learning with STEW for seven iterations, evaluating the algorithm after iterations 1, 3, and 7. 
`python run_stew_test.py`

Other regularization terms can be tested by setting the `regularization` parameter to `"ridge"`, `"nonnegative"`, `"ols"` (= no regularization), or `"ew"` (equal weights).








