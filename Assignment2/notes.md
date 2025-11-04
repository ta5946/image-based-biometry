## Human-Driven Binary Image Features distance

_Equal error rate 0.11% at similarity threshold 0.64_

- Program some extraction method (by default that is
LBP based on the paper: https://ieeexplore.ieee.org/document/1017623) within IrisRecognition.py,
function `extractIBBCode`. Make sure to find and set
optimal parameters.

- Program comparison of computed codes within
IrisRecognition.py, function `matchIBBCodes`.

- List the core (LBP) improvements (parts you implemented/optimized) in the comments on top of the
`extractIBBCode` function.


## Local Binary Patterns distance

- Basic, parameters `R=1`, `P=8`

_Equal error rate 9.67% at similarity threshold 0.58_

- Basic, parameters `R=2`, `P=16`

_Equal error rate 9.07% at similarity threshold 0.59_

- Basic, parameters `R=3`, `P=24`

_Equal error rate 7.84% at similarity threshold 0.59_

- Basic, parameters `R=5`, `P=40`

_Equal error rate 6.80% at similarity threshold 0.61_

- **Uniform** LBP did not improve the accuracy!

- Basic with **CLAHE**, parameters `R=3`, `P=24`

_Equal error rate 6.58% at similarity threshold 0.56_
