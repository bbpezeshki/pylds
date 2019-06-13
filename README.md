# Forked Version of PyLDS from Matt Johnson
2019-06/13<br/>
pylds was forked from Matt Johnson's repository for use in constructing structured VAE on linear dynamical systems.
Initial testing to be done on 1D moving dot data ("pong"-like linear dynamical system).
<br/>
Dependencies:<br/>
    pybasicbayes (fork available here: https://github.com/bbpezeshki/pybasicbayes.git, forked from Matt Johnson)


# PyLDS: Bayesian inference for linear dynamical systems [![Test status](https://travis-ci.org/mattjj/pylds.svg?branch=master)](https://travis-ci.org/mattjj/pylds)
_Authors: Matt Johnson and Scott Linderman_

This package supports Bayesian learning and inference  via Gibbs sampling,
structured mean field, and expectation maximization (EM) for
dynamical systems with linear Gaussian state dynamics and
either linear Gaussian or count observations. For count data,
we support either Pólya-gamma augmentation or Laplace approximation.
All inference algorithms benefit from fast message passing code
written in Cython with direct calls to the BLAS and LAPACK routines
linked to the scipy build.

# Installation

To install from pypi, just run

```
pip install pylds
```

To install from a clone of the git repository, you need to install Cython.
Here's one way to do it:

```
pip install cython
git clone https://github.com/mattjj/pylds.git
pip install -e pylds
```

To handle count data, you'll also need
[pypolyagamma](https://github.com/slinderman/pypolyagamma), which can be
installed with

```
pip install pypolyagamma>=1.1
```

# Example
PyLDS exposes a variety of classes and functions for working with linear
dynamical systems. For example, the following snippet will
generate synthetic data from a random model:
```python
import numpy.random as npr
from pylds.models import DefaultLDS

D_obs = 1       # Observed data dimension
D_latent = 2	# Latent state dimension
D_input = 0	    # Exogenous input dimension
T = 2000  	    # Number of time steps to simulate

true_model = DefaultLDS(D_obs, D_latent, D_input)
inputs = npr.randn(T, D_input)
data, stateseq = true_model.generate(T, inputs=inputs)

# Compute the log likelihood of the data with the true params
true_ll = true_model.log_likelihood() 
```
The `DefaultLDS` constructor initializes an LDS with a
random rotational dynamics matrix. The outputs are `data`, a `T x D_obs` 
matrix of observations, and `stateseq`, a `T x D_latent` matrix
of latent states.

Now create another LDS and try to infer the latent states and
learn the parameters given the observed data. 

```python
# Create a separate model and add the observed data
test_model = DefaultLDS(D_obs, D_latent, D_input)
test_model.add_data(data)

# Run the Gibbs sampler
N_samples = 100
def update(model):
    model.resample_model()
    return model.log_likelihood()

lls = [update(test_model) for _ in range(N_samples)]
```

We can plot the log likelihood over iterations to assess the
convergence of the sampling algorithm:

```python
# Plot the log likelihoods
plt.figure()
plt.plot([0, N_samples], true_ll * np.ones(2), '--k', label="true")
plt.plot(np.arange(N_samples), lls, color=colors[0], label="test")
plt.xlabel("iteration")
plt.ylabel("training likelihood")
plt.legend(loc="lower right")
```
![Log Likelihood](aux/demo_ll.png)

We can also smooth the observations with the test model.
```python
# Smooth the data
smoothed_data = test_model.smooth(data, inputs)

plt.figure()
plt.plot(data, color=colors[0], lw=2, label="observed")
plt.plot(smoothed_data, color=colors[1], lw=1, label="smoothed")
plt.xlabel("Time")
plt.xlim(0, 500)
plt.ylabel("Smoothed Data")
plt.legend(loc="upper center", ncol=2)
```

![Smoothed Data](aux/demo_smooth.png)

Check out the [examples](/examples) directory for demos of other
types of inference, as well as examples of how to work with count
data and missing observations.

For a lower-level interface to the fast message passing functions,
see [lds_messages.pyx](pylds/lds_messages.pyx),
[lds_info_messages.pyx](pylds/lds_info_messages.pyx), and
[lds_messages_interface.py](pylds/lds_messages_interface.py).


