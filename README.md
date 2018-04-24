# pygsl-div
Python implementation of GSL-div as described in "An information theoretic criterion for empirical validation of simulation models" by Francesco Lamperti. http://dx.doi.org/10.1016/j.ecosta.2017.01.006

### How To
#### CLI using csv files that contain the time-series

`python pygsl_div.py` - to use with defaults

`python pygsl_div.py --help` - for help and input description

Paper example 1: `python pygsl_div.py --b 3 --l 3 --min_per 0 --max_per 100 --state_space "(0, 1)"`

#### In any script by importing it as a module

Paper example 1:

```
import pygsl_div
import numpy as np

x_t = [0.2, 0.3, 0.8, 0.4, 0.45, 0.15]
x_t_a = [0.1, 0.25, 0.72, 0.45, 0.5, 0.35]

res = pygsl_div.gsl_div(np.array([x_t]), np.array([x_t_a]*2),
                    'add-progressive', b=3, L=3, min_per=0, max_per=100,
                    state_space=(0, 1))

print(res)
```
