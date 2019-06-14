from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

import scipy.stats

from pybasicbayes.util.text import progprint_xrange

from pylds.models import DefaultLDS

import sys
import pprint as pp
pp = pp.PrettyPrinter(indent=4)

#npr.seed(0)

# Set parameters
## Pong Initialization and Dynamics
Start = 0  # start position in cycle [0,1) (0 = start of cyle at baseline position going up, 0.5 = half way through cycle back at baselineposition going down)
Angle = 7
numPixels = 20  # number of pixels ball bounces back and forth between
PositionNoise = 0 # stdv for normally adding noise for inacurate position measurements
Blur = 1.25		# stdv for blurring ball image
WhiteNoise = 0.05 # stdv for video noise
assert(Start >= 0 and Start < 1)
## Data Parameters
N_obs = 1  #number of independent measurements at each timepoint
D_obs = N_obs * numPixels
D_latent = 2 * D_obs
T = 100

T_given = 25  #int(0.8*T)
T_predict = T-T_given
Per = Angle*4 #number of timepoints to complete a cylce

def calcRelativeCyclePosition(t,Start,Per):
	absoluteCyclePosition = t/Per + Start
	relativeCyclePosition = absoluteCyclePosition - int(absoluteCyclePosition)
	return relativeCyclePosition

def calcPongPosition(t, Start, Per, numPixels):

	relativeCyclePosition = calcRelativeCyclePosition(t,Start,Per)
	Amp = (numPixels-1) / 2
	pongPositionRelativeToBaseline = None
	if relativeCyclePosition < 0.25:
		pongPositionRelativeToBaseline = relativeCyclePosition * 4 * Amp
	elif relativeCyclePosition < 0.5:
		pongPositionRelativeToBaseline = Amp - (relativeCyclePosition - 0.25) * 4 * Amp
	elif relativeCyclePosition < 0.75:
		pongPositionRelativeToBaseline = (relativeCyclePosition - 0.5) * -4 * Amp
	elif relativeCyclePosition < 1:
		pongPositionRelativeToBaseline = -Amp + (relativeCyclePosition - 0.75) * 4 * Amp
	pixelPosition = pongPositionRelativeToBaseline + Amp

	#pp.pprint(pixelPosition)

	return pixelPosition


def createExactPongLocations(T, N_obs, Start, Per, numPixels):
	exactLocations = np.empty((T,N_obs))
	for t in range(T):
		exactLocations[t] = calcPongPosition(t, Start, Per, numPixels) * N_obs

	#pp.pprint(exactLocations)

	return exactLocations;

def createNoisyPongLocations(T, N_obs, Start, Per, numPixels, PositionNoise):
	exactLocations = createExactPongLocations(T, N_obs, Start, Per, numPixels)
	noise = np.random.normal(0,PositionNoise,exactLocations.shape)
	noisyLocations = exactLocations + noise

	#pp.pprint(noisyLocations)

	return noisyLocations

def createDataWithWhiteNoise(WhiteNoise,data):
	whiteNoise = np.random.normal(0.15,WhiteNoise,data.shape)
	dataWithWhiteNoise = data + np.absolute(whiteNoise)

	#pp.pprint(dataWithWhiteNoise)

	return dataWithWhiteNoise;

def createNoisyPongVideoData(T, N_obs, Start, Per, numPixels, PositionNoise, Blur, WhiteNoise):
	noisyLocations = createNoisyPongLocations(T, N_obs, Start, Per, numPixels, PositionNoise)
	noisyVideos = np.empty((T,N_obs,numPixels))
	for t,locationMeasurements in enumerate(noisyLocations):
		for m,loc in enumerate(locationMeasurements):
			n = int(Blur*250); #number of samples to use to generate blurring
			samples = np.random.normal(loc,Blur,n)
			bins = np.arange(numPixels+1) - 0.5
			unnormalizedPixels = np.histogram(samples,bins)[0]
			pixelNormalizer = (scipy.stats.norm(0, Blur).cdf(0.5+Blur**2*0.01) - scipy.stats.norm(0, Blur).cdf(-0.5-Blur**2*0.01)) * (n)
			normalizedPixels = np.clip( (unnormalizedPixels / pixelNormalizer)*0.5, 0, 0.95)
			noisyVideos[t][m] = normalizedPixels
	rawSensorDataWithWhiteNoise = createDataWithWhiteNoise(WhiteNoise, noisyVideos)
	noisyVideosWithWhiteNoise = np.clip( rawSensorDataWithWhiteNoise , 0, 1)
	#np.set_printoptions(threshold=sys.maxsize)
	#pp.pprint(noisyVideosWithWhiteNoise)
	return noisyVideosWithWhiteNoise



videos = createNoisyPongVideoData(T, N_obs, Start, Per, numPixels, PositionNoise, Blur, WhiteNoise)
data = videos.reshape(T,D_obs)

#np.set_printoptions(threshold=sys.maxsize)
#pp.pprint(videos)
#pp.pprint(videos.shape)
#print()
#pp.pprint(data)
#pp.pprint(data.shape)

# Fit with another LDS
model = DefaultLDS(D_obs, D_latent)
model.add_data(data)

# Initialize with a few iterations of Gibbs
for _ in progprint_xrange(40): #previously 10
    model.resample_model()

# Run EM
def update(model):
    vlb = model.meanfield_coordinate_descent_step()
    return vlb

vlbs = [update(model) for _ in progprint_xrange(200)]  #previously 50

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



#plot video frames


lowerBoundToTruncateVideos = max(0, T_given - 200)
xRange = (lowerBoundToTruncateVideos, T)

averageVideo = np.mean(videos,axis=1)
#print(averageVideo.shape)
#print(averageVideo[0])
#print(averageVideo[-1])

truncatedAverageVideo = averageVideo[lowerBoundToTruncateVideos:]
ballPositions = [np.argmax(t.mean(axis=0)) for t in videos]

givenAverageVideo = averageVideo[:T_given]
#print(givenVideos.shape)

predVideos = preds.reshape(T_predict,N_obs,numPixels)
averagePredictedVideo = np.mean(predVideos,axis=1)
#print(averagePredictedVideo.shape)
print(averagePredictedVideo[0])
print(averagePredictedVideo[-1])

shift = np.array([np.amin(averagePredictedVideo, axis=1)])
shiftedAveragePredictedVideo = averagePredictedVideo - shift.transpose()
normalizingValues = np.array([np.amax(shiftedAveragePredictedVideo)])
normalizedAveragePredictedVideo = shiftedAveragePredictedVideo / normalizingValues.transpose()
#print(normalizedAveragePredictedVideo.shape)
print(normalizedAveragePredictedVideo[0])
print(normalizedAveragePredictedVideo[-1])

predictedBallPositions = [np.argmax(t.mean(axis=0)) for t in predVideos]

averageGivenAndPredictedVideo = np.vstack((givenAverageVideo,normalizedAveragePredictedVideo))
#print(givenAndPredictedVideos.shape)



truncatedAverageGivenAndPredictedVideo = averageGivenAndPredictedVideo[lowerBoundToTruncateVideos:]
#print(truncatedAverageGivenAndPredictedVideo.shape)

plt.figure("orignal")
plt.imshow(truncatedAverageVideo.transpose(),cmap="gray")
plt.show()

plt.figure()
plt.imshow(truncatedAverageGivenAndPredictedVideo.transpose(),cmap="gray")
plt.show()


#calculate most probable location of ball
ballPositions = [np.argmax(t.mean(axis=0)) for t in videos]
predVideos = preds.reshape(T_predict,N_obs,numPixels)
predictedBallPositions = [np.argmax(t.mean(axis=0)) for t in predVideos]

# Plot the predictions
plt.figure()
plt.plot(np.arange(T), ballPositions, 'b-', label="true")
plt.plot(T_given + np.arange(T_predict), predictedBallPositions, 'r--', label="prediction")
ylim = plt.ylim()
plt.plot([T_given, T_given], ylim, '-k')
plt.xlabel('time index')
plt.xlim(max(0, T_given - 200), T)
plt.ylabel('prediction')
plt.ylim(ylim)
plt.legend()

plt.show()
