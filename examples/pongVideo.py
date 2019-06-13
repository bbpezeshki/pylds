from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from scipy.signal import sawtooth

from pybasicbayes.util.text import progprint_xrange

from pylds.models import DefaultLDS

import pprint as pp
pp = pp.PrettyPrinter(indent=4)

#npr.seed(0)

# Set parameters
## Data Parameters
D_obs = 20
D_latent = 10
T = 2000
T_given = 1800
T_predict = 200

### data generation

triangle = lambda t: sawtooth(np.pi*t, width=0.5)
make_dot_trajectory = lambda x0, v: lambda t: triangle(v*(t + (1+x0)/2.))
make_renderer = lambda grid, sigma: lambda x: np.exp(-1./2 * (x - grid)**2/sigma**2)

#render_sigma = amplitude, 
def make_dot_data(image_width, T, num_steps=T, x0=0.0, v=0.5, render_sigma=0.2, noise_sigma=0.1):
    grid = np.linspace(-1, 1, image_width, endpoint=True)
    render = make_renderer(grid, render_sigma)
    x = make_dot_trajectory(x0, v)
    images = np.vstack([render(x(t)) for t in np.linspace(0, T, num_steps)])
    return images + noise_sigma * npr.randn(*images.shape)

data = make_dot_data(D_obs, T)

#pp.pprint(data)
#pp.pprint(data.shape)

# Fit with another LDS
model = DefaultLDS(D_obs, D_latent)
model.add_data(data)

# Initialize with a few iterations of Gibbs
for _ in progprint_xrange(10):
    model.resample_model()

# Run EM
def update(model):
    vlb = model.meanfield_coordinate_descent_step()
    return vlb

vlbs = [update(model) for _ in progprint_xrange(50)]

# Sample from the mean field posterior
model.resample_from_mf()

# Plot the log likelihoods
plt.figure()
plt.plot(vlbs)
plt.xlabel('iteration')
plt.ylabel('variational lower bound')

# Predict forward in time
given_data= data[:T_given]

preds = \
    model.sample_predictions(given_data, Tpred=T_predict)

#calculate most probable location of ball
pixels = np.arange(D_obs)
data = [sum(x*pixels) for x in data]
preds = [sum(x*pixels) for x in preds]

# Plot the predictions
plt.figure()
plt.plot(np.arange(T), data, 'b-', label="true")
plt.plot(T_given + np.arange(T_predict), preds, 'r--', label="prediction")
ylim = plt.ylim()
plt.plot([T_given, T_given], ylim, '-k')
plt.xlabel('time index')
plt.xlim(max(0, T_given - 200), T)
plt.ylabel('prediction')
plt.ylim(ylim)
plt.legend()

plt.show()
