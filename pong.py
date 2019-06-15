from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from pybasicbayes.util.text import progprint_xrange

from pylds.models import DefaultLDS

#import sys
#import pprint as pp
#pp = pp.PrettyPrinter(indent=4)

#npr.seed(0)

# Set parameters
## Data Parameters
D_obs = 1 # number of independent measurements of ball location
D_latent = 2
T = 1000
T_given = 600
T_predict = 400
## Pong Initialization and Dynamics
Start = 0  # start position in cycle [0,1) (0 = start of cyle at baseline position going up, 0.5 = half way through cycle back at baselineposition going down)
Per = 50  # time steps for one cycle completion
Amp = 2
PositionNoise = 0 # stdv for normally adding noise for inacurate position measurements (0.1 recommended for noise)
assert(Start >= 0 and Start < 1)

def calcRelativeCyclePosition(t,Start,Per):
	absoluteCyclePosition = t/Per + Start
	relativeCyclePosition = absoluteCyclePosition - int(absoluteCyclePosition)
	return relativeCyclePosition

def calcDotPosition(t, Start, Per, Amp):
	relativeCyclePosition = calcRelativeCyclePosition(t,Start,Per)
	dotPosition = None
	if relativeCyclePosition < 0.25:
		dotPosition = relativeCyclePosition * 4 * Amp
	elif relativeCyclePosition < 0.5:
		dotPosition = Amp - (relativeCyclePosition - 0.25) * 4 * Amp
	elif relativeCyclePosition < 0.75:
		dotPosition = (relativeCyclePosition - 0.5) * -4 * Amp
	elif relativeCyclePosition < 1:
		dotPosition = -Amp + (relativeCyclePosition - 0.75) * 4 * Amp
	return dotPosition


def createExactPongData(T, D_obs, Start, Per, Amp):
	exactData = np.empty((T,D_obs))
	for t in range(T):
		exactData[t] = [calcDotPosition(t, Start, Per, Amp)] * D_obs
	return exactData;

def createNoisyPongData(T, D_obs, Start, Per, Amp, PositionNoise):
	exactData = createExactPongData(T, D_obs, Start, Per, Amp)
	noise = np.random.normal(0,PositionNoise,exactData.shape)
	noisyData = exactData + noise
	return noisyData


data = createNoisyPongData(T, D_obs, Start, Per, Amp, PositionNoise)

#np.set_printoptions(threshold=sys.maxsize)
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

vlbs = [update(model) for _ in progprint_xrange(2000)]

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
data = data.mean(axis=1)
preds = preds.mean(axis=1)

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
