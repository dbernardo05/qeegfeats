
import sys
import mne
import numpy as np
import scipy as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from pyriemann.estimation import Shrinkage
from pyriemann.tangentspace import TangentSpace

from preproc.utils.transformer import PFD, HFD, Hurst
from pyriemann.utils.covariance import cospectrum

from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
	with open(os.devnull, "w") as devnull:
		old_stdout = sys.stdout
		sys.stdout = devnull
		try:  
			yield
		finally:
			sys.stdout = old_stdout

def _nextpow2(i):
	"""Find next power of 2."""
	n = 1
	while n < i:
		n *= 2
	return n

def mad(data):
	"""Median absolute deviation"""
	m = np.median(np.abs(data - np.median(data)))
	return m


class Filterer(BaseEstimator, TransformerMixin):
	"""Filter."""
	def __init__(self, hp=0.1, lp=20, srate=256, new_srate=40):
		"""Init."""
		self.hp = hp
		self.lp = lp
		self.srate = srate
		self.new_srate = new_srate
		pass

	def fit(self, X, y=None):
		"""
		Fit, do nothing
		"""
		return self

	def bp_filter(self, eeg_arr):
		with suppress_stdout():
			n_channels = eeg_arr.shape[0]
			info = mne.create_info(n_channels, 
				sfreq=self.srate, 
				ch_types = ['eeg']*n_channels, 
				verbose=False)
			raw = mne.io.RawArray(eeg_arr, info)
			raw.filter(self.hp, self.lp, verbose='ERROR', fir_design='firwin')
			raw.resample(self.new_srate, npad='auto')

		return raw.get_data()

	def transform(self, X):
		out = []
		for x in X:
			out.append(self.bp_filter(x))
		return np.array(out)

class Windower(BaseEstimator, TransformerMixin):
	"""Window."""

	def __init__(self, window=60, overlap=0, srate=256):
		"""Init."""
		self.window = window
		self.overlap = overlap
		self.srate = srate
		pass

	def fit(self, X, y=None):
		"""
		Fit, do nothing
		"""
		return self

	def transform(self, X):
		"""
		Detect and remove dropped.
		"""
		if 0: # Debugging
			print('windower shapes:')
			for x in X:
				print(x.shape)
			print('window:', self.window)
			print('overlap:', self.overlap)
			print('srate:', self.srate)
		wi = int(self.window * self.srate)
		ov = int(self.overlap * wi)
		out = []
		for x in X:
			nSamples = x.shape[1]
			ind = list(range(0, nSamples - wi + 1, wi - ov))
			for idx in ind:
				sl = slice(idx, idx + wi)
				out.append(x[:, sl])
		return np.array(out)

class RawContinuous(BaseEstimator, TransformerMixin):
	def __init__(self):
		"""Init."""
		pass

	def fit(self, X, y=None):
		"""
		Fit, do nothing
		"""
		return self

	def transform(self, X):
		"""
		Detect and remove dropped.
		"""
		return np.array(X)

class MinMax(BaseEstimator, TransformerMixin):
	"""Withening."""

	def __init__(self):
		"""Init."""
		pass

	def fit(self, X, y=None):
		"""
		Fit, do nothing
		"""
		return self

	def transform(self, X):
		"""
		Detect and remove dropped.
		"""
		out = []
		for x in X:
			tmp = [np.min(x), np.max(x)]
			out.append(tmp)
		return np.array(out)


from pyriemann.utils.base import invsqrtm
class Whitening(BaseEstimator, TransformerMixin):
	"""Withening."""

	def __init__(self):
		"""Init."""
		pass

	def fit(self, X, y=None):
		"""
		Fit, do nothing
		"""
		return self

	def transform(self, X):
		"""
		Detect and remove dropped.
		"""
		out = []
		for x in X:
			if np.sum(x) != 0:
				cov = np.cov(x)
				W = invsqrtm(cov)
				tmp = np.dot(W.T, x)
			else:
				tmp = x
			out.append(tmp)
		return np.array(out)

from sklearn.decomposition import PCA
class ApplyPCA(BaseEstimator, TransformerMixin):
	"""Withening."""

	def __init__(self, n_components=2):
		"""Init."""
		self.n_components = n_components
		pass

	def fit(self, X, y=None):
		"""
		Fit, do nothing
		"""
		return self

	def transform(self, X):
		"""
		Detect and remove dropped.
		"""
		out = []
		for x in X:
			tmp = PCA(self.n_components).fit_transform(x.T).T
			out.append(tmp)
		return np.array(out)

class Slicer(BaseEstimator, TransformerMixin):
	"""Window."""

	def __init__(self, tmin=0, tmax=60, srate=200):
		"""Init."""
		self.tmin = tmin
		self.tmax = tmax
		self.srate = srate
		pass

	def fit(self, X, y=None):
		"""
		Fit, do nothing
		"""
		return self

	def transform(self, X):
		"""
		Detect and remove dropped.
		"""
		tmin = int(self.tmin * self.srate)
		tmax = int(self.tmax * self.srate)
		sl = slice(tmin, tmax)
		out = []
		for x in X:
			out.append(x[:, sl])
		return np.array(out)

class RemoveDropped(BaseEstimator, TransformerMixin):
	"""Remove dropped packet."""

	def __init__(self):
		"""Init."""
		pass

	def fit(self, X, y=None):
		"""
		Fit, do nothing
		"""
		return self

	def transform(self, X):
		"""
		Detect and remove dropped.
		"""
		out = []
		for x in X:
			good_idx = (np.sum(x**2, 0) != 0)
			if np.sum(good_idx)==120000:  # change from 240000 to 120000 (10*60*200)
				# if data only contains dropped sample, pass it as it
				# to avoid passing empty array
				out.append(x)
			else:
				# else remove dropped packet
				out.append(x[:, good_idx])
		return np.array(out)


class IsEmpty(BaseEstimator, TransformerMixin):
	"""Is the data empty ?"""

	def __init__(self):
		"""Init."""
		pass

	def fit(self, X, y=None):
		"""
		Fit, do nothing
		"""
		return self

	def transform(self, X):
		"""
		Detect and remove dropped.
		"""
		out = []
		for x in X:
			if np.sum(x) == 0:
				# if data only contains dropped sample, pass it as it
				# to avoid passing empty array
				out.append([1])
			else:
				# else remove dropped packet
				out.append([0])
		return np.array(out)

class InterpolateSpikes(BaseEstimator, TransformerMixin):
	"""Remove dropped packet."""

	def __init__(self,th=20):
		"""Init."""
		self.th = th

	def fit(self, X, y=None):
		"""
		Fit, do nothing
		"""
		return self

	def transform(self, X):
		"""
		Detect and rinerpolates dropped sample
		"""
		out = []
		for x in X:
			avg_ref =  np.mean(x, 0)
			m = mad(avg_ref)
			no_spikes = avg_ref < (self.th * m)
			#print (np.sum(no_spikes), m)
			if m!=0:
				indices = np.arange(len(avg_ref))
				for ii, ch in enumerate(x):
					x[ii] = np.interp(indices, indices[no_spikes], ch[no_spikes])
			out.append(x)
		return np.array(out)

class Useless(BaseEstimator, TransformerMixin):
	"""Remove dropped packet."""

	def __init__(self, nsamples=2):
		"""Init."""
		self.nsamples = nsamples

	def fit(self, X, y=None):
		"""
		Fit, do nothing
		"""
		return self

	def transform(self, X):
		"""
		Detect and rinerpolates dropped sample
		"""
		out = []
		for x in X:
			tmp = x[:, 0:self.nsamples].flatten()
			out.append(tmp)
		return np.array(out)
#### coherence

from scipy.signal import filtfilt, butter

class FrequenctialFilter(BaseEstimator, TransformerMixin):
	"""Withening."""

	def __init__(self, order=4, freqs=[4, 15], ftype='bandpass'):
		"""Init."""
		self.order = order
		self.freqs = freqs
		self.ftype = ftype

	def fit(self, X, y=None):
		"""
		Fit, do nothing
		"""
		return self

	def transform(self, X):
		"""
		Detect and remove dropped.
		"""
		b, a = butter(self.order, np.array(self.freqs) / 200., self.ftype)
		out = filtfilt(b, a, X, axis=-1)
		return out


from scipy.signal import hanning, welch

from scipy.signal import argrelextrema

def find_peak(c, fr, order=5, max_peak=3):
	out = []
	for ci in c:
		tmp = []
		for ch in ci.T:
			a = argrelextrema(ch, np.greater, order=order)[0]
			if len(a) < max_peak:
				a = np.r_[a, [0] * (max_peak - len(a))]

			tmp.extend(list(fr[a[0:max_peak]]))
		out.append(np.array(tmp))
	return np.array(out)

def peak_freq(data, window=256, fs=200, overlap=0., ignore_dropped=False,
			   frequencies=[6, 20]):

	nChan, nSamples = data.shape
	noverlap = int(overlap * window)
	windowVals = hanning(window)

	# get the corresponding indices for custom frequencies
	freqs = np.fft.fftfreq(window, d=1./fs)[:window/2]
	idx_freqs = []
	idx_freqs.append((freqs < frequencies[0]) | (freqs > frequencies[1]))

	ind = list(range(0, nSamples - window + 1, window-noverlap))

	numSlices = len(ind)
	slices = range(numSlices)

	Slices = []
	for iSlice in slices:
		thisSlice = data[:, ind[iSlice]:ind[iSlice] + window]
		if np.sum(np.sum(thisSlice**2, axis=0)>0):
			freqs, thisfft = welch(thisSlice, fs=200, nfft=window/2)
			Slices.append(thisfft.T)
	if len(Slices) > 0:
		Slices = np.array(Slices)
		a = find_peak(Slices, freqs, order=5, max_peak=3)
	else:
		a = np.nan
	return a


def slidingFFT(data, window=256, fs=200, overlap=0., ignore_dropped=False,
				frequencies=None, aggregate=True, phase=False):

	nChan, nSamples = data.shape
	noverlap = int(overlap * window)
	windowVals = hanning(window)

	# get the corresponding indices for custom frequencies
	freqs = np.fft.fftfreq(window, d=1./fs)[:window/2]
	idx_freqs = []
	if frequencies is not None:
		for fr in frequencies:
			tmp = (freqs >= fr[0]) & (freqs < fr[1])
			idx_freqs.append(np.where(tmp)[0])
			numFreqs = len(idx_freqs)
	else:
		numFreqs = len(freqs)
	# get the indices of dropped data
	if ignore_dropped:
		dropped = (np.sum(data**2, 0) == 0)

	ind = list(range(0, nSamples - window + 1, window-noverlap))

	numSlices = len(ind)
	slices = range(numSlices)
	Slices = np.zeros((numSlices, numFreqs, nChan), dtype=np.complex_)
	for iSlice in slices:
		sl = slice(ind[iSlice], ind[iSlice] + window)
		if ignore_dropped:
			if np.sum(dropped[sl]) > 0:
				continue

		thisSlice = data[:, sl]
		thisSlice = windowVals*thisSlice
		thisfft = np.fft.fft(thisSlice).T
		if frequencies is None:
			Slices[iSlice] = thisfft[1:(window/2 + 1)]
		else:
			for fr, idx in enumerate(idx_freqs):
				Slices[iSlice, fr, :] = thisfft[idx].mean(0)

	Slices = Slices.transpose(0, 2, 1)
	if aggregate:
		Slices = np.concatenate(Slices.transpose(1, 2, 0), axis=0)
	else:
		Slices = Slices.transpose(2, 1, 0)

	if phase:
		Slices = np.arctan2(np.imag(Slices), np.real(Slices))
	else:
		Slices = np.abs(Slices)

	return Slices


class SlidingFFT(BaseEstimator, TransformerMixin):

	"""Slinding FFT
	"""

	def __init__(self, window=256, overlap=0.5, fs=200,
				 frequencies=None, aggregate=True, ignore_dropped=False,
				 phase=False):
		"""Init."""
		self.window = window
		self.overlap = overlap
		self.fs = fs
		self.frequencies = frequencies
		self.aggregate = aggregate
		self.ignore_dropped = ignore_dropped
		self.phase = phase

	def fit(self, X, y=None):
		"""Fit.

		Do nothing. For compatibility purpose.

		Parameters
		----------
		X : ndarray, shape (n_trials, n_channels, n_samples)
			ndarray of trials.
		y : ndarray shape (n_trials,)
			labels corresponding to each trial, not used.

		Returns
		-------
		self : CospCovariances instance
			The CospCovariances instance.
		"""
		return self

	def transform(self, X):
		"""Estimate the cospectral covariance matrices.

		Parameters
		----------
		X : ndarray, shape (n_trials, n_channels, n_samples)
			ndarray of trials.

		Returns
		-------
		covmats : ndarray, shape (n_trials, n_channels, n_channels, n_freq)
			ndarray of covariance matrices for each trials and for each
			frequency bin.
		"""
		Nt, Ne, _ = X.shape
		out = []

		for i in range(Nt):
			S = slidingFFT(X[i], window=self.window, fs=self.fs,
							overlap=self.overlap, frequencies=self.frequencies,
							aggregate=self.aggregate, phase=self.phase,
							ignore_dropped=self.ignore_dropped)
			out.append(S)

		return S


def coherences(data, window=256, fs=200, overlap=0., ignore_dropped=False,
			   frequencies=[[0.1, 4], [4, 8], [8, 15], [15, 30], [30, 90]], # , [90, 170]
			   aggregate=False, transpose=False, normalize=True, return_global_median=False):

	nChan, nSamples = data.shape

	noverlap = int(overlap * window)
	windowVals = hanning(window)

	# get the corresponding indices for custom frequencies
	freqs = np.fft.fftfreq(window, d=1./fs)[:int(window/2)]
	idx_freqs = []
	if frequencies is not None:
		for fr in frequencies:
			tmp = (freqs >= fr[0]) & (freqs < fr[1])
			idx_freqs.append(np.where(tmp)[0])
			numFreqs = len(idx_freqs)
	else:
		numFreqs = len(freqs)
	# get the indices of dropped data
	if ignore_dropped:
		dropped = (np.sum(data**2, 0) == 0)

	ind = list(range(0, nSamples - window + 1, window-noverlap))

	numSlices = len(ind)
	FFTSlices = {}
	FFTConjSlices = {}
	Pxx = {}
	slices = range(numSlices)
	normVal = np.linalg.norm(windowVals)**2

	Slices = np.zeros((numSlices, numFreqs, nChan), dtype=np.complex_)
	for iSlice in slices:
		thisSlice = data[:, ind[iSlice]:ind[iSlice] + window]
		#if sum(thisSlice)!=0:
		thisSlice = windowVals*thisSlice
		thisfft = np.fft.fft(thisSlice).T
		if frequencies is None:
			Slices[iSlice] = thisfft[0:window/2]
		else:
			for fr, idx in enumerate(idx_freqs):
				Slices[iSlice, fr, :] = thisfft[idx].mean(0)

	if transpose:
		Slices = Slices.transpose(0, 2, 1)
		numFreqs = 20 #21 #16

	if aggregate:
		Slices = np.concatenate(Slices.transpose(1,2,0), axis=0)
		Slices = np.atleast_3d(Slices).transpose(1,2,0)
		numFreqs = 1

	FFTConjSlices = np.conjugate(Slices)
	Pxx = np.divide(np.mean(abs(Slices)**2, axis=0), normVal)
	del ind, windowVals

	Cxy = []
	for fr in range(numFreqs):
		Pxy = np.dot(Slices[:, fr].T, FFTConjSlices[:, fr]) / normVal
		Pxy /= len(Slices)
		if normalize:
			Pxxx = np.outer(Pxx[fr], Pxx[fr])
			Cxy.append(abs(Pxy)**2 / Pxxx)
		else:
			Cxy.append(abs(Pxy)**2)

	if return_global_median:
		# Compute the median coherence across all channel pairs, ignoring NaNs
		Cxy = np.array(Cxy)
		Cxy_median = np.nanmedian(Cxy, axis=(1,2))
		Cxy_median = np.expand_dims(Cxy_median, axis=0)
		return Cxy_median

	else:

		return np.array(Cxy).transpose((1, 2, 0))

class Coherences(BaseEstimator, TransformerMixin):

	"""Estimation of cospectral covariance matrix.

	Covariance estimation in the frequency domain. this method will return a
	4-d array with a covariance matrice estimation for each trial and in each
	frequency bin of the FFT.

	Parameters
	----------
	window : int (default 128)
		The lengt of the FFT window used for spectral estimation.
	overlap : float (default 0.75)
		The percentage of overlap between window.
	fmin : float | None , (default None)
		the minimal frequency to be returned.
	fmax : float | None , (default None)
		The maximal frequency to be returned.
	fs : float | None, (default None)
		The sampling frequency of the signal.

	See Also
	--------
	Covariances
	HankelCovariances
	Coherences
	"""

	def __init__(self, window=256, overlap=0.5, fs=200,
				 frequencies=[[0.1, 4], [4, 8], [8, 15], [15, 30], [30, 90]], # , [90, 170]
				 aggregate=False, transpose=False, normalize=True, return_global_median=False):
		"""Init."""
		self.window = window
		self.overlap = overlap
		self.fs = fs
		self.frequencies = frequencies
		self.aggregate = aggregate
		self.transpose = transpose
		self.normalize = normalize
		self.return_global_median = return_global_median

	def fit(self, X, y=None):
		"""Fit.

		Do nothing. For compatibility purpose.

		Parameters
		----------
		X : ndarray, shape (n_trials, n_channels, n_samples)
			ndarray of trials.
		y : ndarray shape (n_trials,)
			labels corresponding to each trial, not used.

		Returns
		-------
		self : CospCovariances instance
			The CospCovariances instance.
		"""
		return self

	def transform(self, X):
		"""Estimate the cospectral covariance matrices.

		Parameters
		----------
		X : ndarray, shape (n_trials, n_channels, n_samples)
			ndarray of trials.

		Returns
		-------
		covmats : ndarray, shape (n_trials, n_channels, n_channels, n_freq)
			ndarray of covariance matrices for each trials and for each
			frequency bin.
		"""
		Nt, Ne, _ = X.shape
		out = []

		for i in range(Nt):
			S = coherences(X[i], window=self.window, fs=self.fs,
						   overlap=self.overlap, frequencies=self.frequencies,
						   aggregate=self.aggregate, transpose=self.transpose,
						   normalize=self.normalize, return_global_median=self.return_global_median)
			if np.sum(S)==0:
				S = (np.zeros_like(S) + 1) * np.nan
			out.append(S)

		return np.array(out)

class FlattenedCoherences(Coherences, TransformerMixin):
	
	def transform(self, X):
		"""Estimate the cospectral covariance matrices.

		Parameters
		----------
		X : ndarray, shape (n_trials, n_channels, n_samples)
			ndarray of trials.

		Returns
		-------
		covmats : ndarray, shape (n_trials, n_channels, n_channels, n_freq)
			ndarray of covariance matrices for each trials and for each
			frequency bin.
		"""
		Nt, Ne, _ = X.shape
		out = []

		for i in range(Nt):
			S = coherences(X[i], window=self.window, fs=self.fs,
						   overlap=self.overlap, frequencies=self.frequencies,
						   aggregate=self.aggregate, transpose=self.transpose,
						   normalize=self.normalize)
			if np.sum(S)==0:
				S = (np.zeros_like(S) + 1) * np.nan
			# out.append(S.reshape(S.shape[0], -1))
			out.append(S.reshape(-1)) # flatten to one-dimensional array!
		return np.array(out)


class PeakFreq(BaseEstimator, TransformerMixin):

	"""Estimation of cospectral covariance matrix.

	Covariance estimation in the frequency domain. this method will return a
	4-d array with a covariance matrice estimation for each trial and in each
	frequency bin of the FFT.

	Parameters
	----------
	window : int (default 128)
		The lengt of the FFT window used for spectral estimation.
	overlap : float (default 0.75)
		The percentage of overlap between window.
	fmin : float | None , (default None)
		the minimal frequency to be returned.
	fmax : float | None , (default None)
		The maximal frequency to be returned.
	fs : float | None, (default None)
		The sampling frequency of the signal.

	See Also
	--------
	Covariances
	HankelCovariances
	Coherences
	"""

	def __init__(self, window=256, overlap=0.5, fs=200,
				 frequencies=[[0.1, 4], [4, 8], [8, 15], [15, 30], [30, 90]]):  # , [90, 170]
		"""Init."""
		self.window = window
		self.overlap = overlap
		self.fs = fs
		self.frequencies = frequencies

	def fit(self, X, y=None):
		"""Fit.

		Do nothing. For compatibility purpose.

		Parameters
		----------
		X : ndarray, shape (n_trials, n_channels, n_samples)
			ndarray of trials.
		y : ndarray shape (n_trials,)
			labels corresponding to each trial, not used.

		Returns
		-------
		self : CospCovariances instance
			The CospCovariances instance.
		"""
		return self

	def transform(self, X):
		"""Estimate the cospectral covariance matrices.

		Parameters
		----------
		X : ndarray, shape (n_trials, n_channels, n_samples)
			ndarray of trials.

		Returns
		-------
		covmats : ndarray, shape (n_trials, n_channels, n_channels, n_freq)
			ndarray of covariance matrices for each trials and for each
			frequency bin.
		"""

		out = []

		for x in X:
			S = peak_freq(x, window=self.window, fs=self.fs,
						  overlap=self.overlap, frequencies=self.frequencies)
			out.append(S)

		return out


class GenericTransformer(BaseEstimator, TransformerMixin):
	"""Remove dropped packet."""

	def __init__(self, method=np.mean, nanshape=(21, 1)):
		"""Init."""
		self.method = method
		self.nanshape = nanshape
		pass

	def fit(self, X, y=None):
		"""
		Fit, do nothing
		"""
		return self

	def transform(self, X):
		"""
		Detect and remove dropped.
		"""
		out = []
		for x in X:
			if np.isnan(x).any():
				tmp = np.ones(self.nanshape) * np.nan
			else:
				tmp = self.method(x)
			out.append(tmp)
		return np.array(out)


class BasicStats(BaseEstimator, TransformerMixin):
	"""Remove dropped packet."""

	def __init__(self):
		"""Init."""
		pass

	def fit(self, X, y=None):
		"""
		Fit, do nothing
		"""
		return self

	def transform(self, X):
		"""
		Detect and remove dropped.
		"""
		out = []
		for x in X:
			m = np.mean(x, 1)
			sd = np.std(x, 1)
			ku = sp.stats.kurtosis(x, 1)
			sk = sp.stats.skew(x, 1)
			p90 = np.percentile(x, 90, axis=1)
			p10 = np.percentile(x, 10, axis=1)
			tmp = np.c_[m, sd, ku, sk, p90, p10]
			if np.all(x == 0) or np.max(x) < 1e-15:
				tmp = np.zeros(tmp.shape)
				tmp[:] = np.nan
			out.append(tmp)

		return np.array(out)

class MedianFilteredStats(BaseEstimator, TransformerMixin):
	"""Remove dropped packet."""

	def __init__(self):
		"""Init."""
		pass

	def fit(self, X, y=None):
		"""
		Fit, do nothing
		"""
		return self

	def transform(self, X):
		"""
		Detect and remove dropped.
		"""
		out = []
		for x in X:
			x_med_filt = sp.signal.medfilt(x, (1,3))
			m = np.mean(x_med_filt, 1)
			sd = np.std(x_med_filt, 1)
			ku = sp.stats.kurtosis(x_med_filt, 1)
			sk = sp.stats.skew(x_med_filt, 1)
			p90 = np.percentile(x_med_filt, 90, axis=1)
			p10 = np.percentile(x_med_filt, 10, axis=1)
			tmp = np.c_[m, sd, ku, sk, p90, p10]
			if np.all(x == 0) or np.max(x) < 1e-15:
				tmp[:] = np.nan
			out.append(tmp)
		return np.array(out)

class MedianFilteredStatsKern5(BaseEstimator, TransformerMixin):
	"""Remove dropped packet."""

	def __init__(self):
		"""Init."""
		pass

	def fit(self, X, y=None):
		"""
		Fit, do nothing
		"""
		return self

	def transform(self, X):
		"""
		Detect and remove dropped.
		"""
		out = []
		for x in X:
			x_med_filt = sp.signal.medfilt(x, (1,5))
			m = np.mean(x_med_filt, 1)
			sd = np.std(x_med_filt, 1)
			ku = sp.stats.kurtosis(x_med_filt, 1)
			sk = sp.stats.skew(x_med_filt, 1)
			p90 = np.percentile(x_med_filt, 90, axis=1)
			p10 = np.percentile(x_med_filt, 10, axis=1)
			tmp = np.c_[m, sd, ku, sk, p90, p10]

			out.append(tmp)
		return np.array(out)


class FlattenedStats(BasicStats):
	def transform(self, X):
		"""
		Detect and remove dropped.
		"""
		out = []
		for x in X:
			m = np.mean(x, 1)
			sd = np.std(x, 1)
			ku = sp.stats.kurtosis(x, 1)
			sk = sp.stats.skew(x, 1)
			p90 = np.percentile(x, 90, axis=1)
			p10 = np.percentile(x, 10, axis=1)
			tmp = np.c_[m, sd, ku, sk, p90, p10]

			out.append(tmp.reshape(-1))
		return np.array(out)

from pyriemann.estimation import HankelCovariances

class AutoCorrMat(BaseEstimator, TransformerMixin):
	"""Remove dropped packet."""

	def __init__(self, order=6, subsample=4, eigenmode=True):
		"""Init."""
		self.order = order
		self.subsample = subsample
		self.eigenmode = eigenmode
		pass

	def fit(self, X, y=None):
		"""
		Fit, do nothing
		"""
		return self

	def transform(self, X):
		"""
		Detect and remove dropped.
		"""
		out = []
		hk = HankelCovariances(delays=self.order, estimator=np.corrcoef)
		for x in X:

			tmp = []
			for i, a in enumerate(x):
				cov_matrix = hk.fit_transform(np.atleast_3d(a[::self.subsample]).transpose(0,2,1))[0]
				if self.eigenmode:
					cov_matrix = np.nan_to_num(cov_matrix)
					tmp.append(np.max(np.linalg.eigvals(cov_matrix)))
				else:
					tmp.append(cov_matrix)

			out.append(tmp)
		if self.eigenmode:
			return np.array(out)[..., np.newaxis]
		else:
			return np.array(out).transpose(0,2,3,1)

from statsmodels.tsa.ar_model import AutoReg
import warnings

class ARError(BaseEstimator, TransformerMixin):
	"""Remove dropped packet."""

	def __init__(self, order=4, subsample=4):
		"""Init."""
		self.order = order
		self.subsample = subsample
		pass

	def fit(self, X, y=None):
		"""
		Fit, do nothing
		"""
		return self

	def transform(self, X):
		"""
		Detect and remove dropped.
		"""
		out = []
		for x in X:
			tmp = []
			for a in x:
				with warnings.catch_warnings():
					warnings.simplefilter('ignore')
					ar_mod = AutoReg(a[::self.subsample], lags=self.order)
				ar_res = ar_mod.fit()
				bse = ar_res.bse
				if len(bse)!=(self.order + 1):
					bse = np.array([np.nan] * (self.order + 1))
				tmp.append(bse)
			out.append(tmp)
		return np.array(out)


class VariousFeatures(BaseEstimator, TransformerMixin):
	"""Remove dropped packet."""

	def __init__(self):
		"""Init."""
		pass

	def fit(self, X, y=None):
		"""
		Fit, do nothing
		"""
		return self

	def transform(self, X):
		"""
		Detect and remove dropped.
		"""
		out = []
		for x in X:
			pfd = PFD().apply(x)
			hfd = HFD().apply(x)
			hurst = Hurst().apply(x)

			tmp = np.c_[pfd, hfd, hurst]

			out.append(tmp)
		return np.array(out)
		
def safe_log10(x, eps=1e-10):     
	result = np.where(x > eps, x, -10)     
	np.log10(result, out=result, where=result > 0)     
	return result

def relative_log_power(data, window=256, fs=256, overlap=0.,
					   frequencies = [[0.1, 4], [4, 8], [8, 12], [12, 30], [30, 70], [70, 180]]): # [90, 170]
	noverlap = int(window * overlap)

	freqs, power = welch(data, fs=fs, nperseg=window, noverlap=noverlap)
	out = []
	if frequencies is None:
		out = power
	else:
		for fr in frequencies:
			tmp = (freqs >= fr[0]) & (freqs < fr[1])
			output=(power[:, tmp].mean(1))
			out.append(output)
	out_arr = np.array(out)
	return safe_log10(out_arr / (np.sum(out_arr, 0)+0.000001))



def relative_log_power_averaged(data, window=256, fs=200, overlap=0.,
					   frequencies = [[0.1, 4], [4, 8], [8, 15], [15, 30], [30, 90]]): # [90, 170]
	noverlap = int(window * overlap)

	freqs, power = welch(data, fs=fs, nperseg=window, noverlap=noverlap)
	out = []
	if frequencies is None:
		out = power
	else:
		for fr in frequencies:
			tmp = (freqs >= fr[0]) & (freqs < fr[1])
			output=np.average(power[:, tmp].mean(1))
			out.append(output)
	out_arr = np.array(out)
	return safe_log10(out_arr / np.sum(out_arr, 0))


def cumulative_log_power(data, window=256, fs=200, overlap=0.):
	noverlap = int(window * overlap)
	freqs, power = welch(data, fs=fs, nperseg=window, noverlap=noverlap)
	out = np.cumsum((power), 1)
	return out / np.atleast_2d(out[:, -1]).T

def spectral_edge_frequency(data, window=256, fs=200, overlap=0., edges=[0.5, 0.7, 0.8, 0.9, 0.95]):
	noverlap = int(window * overlap)
	freqs, power = welch(data, fs=fs, nperseg=window, noverlap=noverlap)
	out = np.cumsum((power), 1)
	out = out / np.atleast_2d(out[:, -1]).T
	ret = []
	if np.sum(np.isnan(out))>0:
		ret = np.ones((len(edges), 21)) * np.nan
	else:
		for edge in edges:
			tmp = []
			for ch in out:
				tmp.append(freqs[np.where(ch>edge)[0][0]])
			ret.append(tmp)
		ret = np.array(ret)
	return ret

class RelativeLogPower(BaseEstimator, TransformerMixin):

	"""Relative power
	"""

	def __init__(self, window=256, overlap=0.5, fs=200, averageAcrossChans=False,
				 frequencies=[[0.1, 4], [4, 8], [8, 15], [15, 30], [30, 90]]): # , [90, 170]
		"""Init."""
		self.window = window
		self.overlap = overlap
		self.fs = fs
		self.averageAcrossChans = averageAcrossChans
		self.frequencies = frequencies

	def fit(self, X, y=None):
		"""Fit.

		Do nothing. For compatibility purpose.

		Parameters
		----------
		X : ndarray, shape (n_trials, n_channels, n_samples)
			ndarray of trials.
		y : ndarray shape (n_trials,)
			labels corresponding to each trial, not used.

		Returns
		-------
		self : CospCovariances instance
			The CospCovariances instance.
		"""
		return self

	def transform(self, X):
		"""Estimate the cospectral covariance matrices.

		Parameters
		----------
		X : ndarray, shape (n_trials, n_channels, n_samples)
			ndarray of trials.

		Returns
		-------
		covmats : ndarray, shape (n_trials, n_channels, n_channels, n_freq)
			ndarray of covariance matrices for each trials and for each
			frequency bin.
		"""
		Nt, Ne, _ = X.shape
		out = []

		for i in range(Nt):
			if self.averageAcrossChans:
				S = relative_log_power_averaged(X[i], window=self.window, fs=self.fs,
							   overlap=self.overlap, frequencies=self.frequencies)
			else:
				S = relative_log_power(X[i], window=self.window, fs=self.fs,
							   overlap=self.overlap, frequencies=self.frequencies)
			out.append(S.T)

		return np.array(out)




class CumulativeLogPower(BaseEstimator, TransformerMixin):

	"""Relative power
	"""

	def __init__(self, window=256, overlap=0.5, fs=200):
		"""Init."""
		self.window = window
		self.overlap = overlap
		self.fs = fs

	def fit(self, X, y=None):
		"""Fit.

		Do nothing. For compatibility purpose.

		Parameters
		----------
		X : ndarray, shape (n_trials, n_channels, n_samples)
			ndarray of trials.
		y : ndarray shape (n_trials,)
			labels corresponding to each trial, not used.

		Returns
		-------
		self : CospCovariances instance
			The CospCovariances instance.
		"""
		return self

	def transform(self, X):
		"""Estimate the cospectral covariance matrices.

		Parameters
		----------
		X : ndarray, shape (n_trials, n_channels, n_samples)
			ndarray of trials.

		Returns
		-------
		covmats : ndarray, shape (n_trials, n_channels, n_channels, n_freq)
			ndarray of covariance matrices for each trials and for each
			frequency bin.
		"""
		Nt, Ne, _ = X.shape
		out = []

		for i in range(Nt):
			S = cumulative_log_power(X[i], window=self.window, fs=self.fs,
						   overlap=self.overlap)
			out.append(S)

		return np.array(out)

class SpectralEdgeFrequency(BaseEstimator, TransformerMixin):

	"""Relative power
	"""

	def __init__(self, window=256, overlap=0.5, fs=200, edges=[0.5, 0.7, 0.8, 0.9, 0.95]):
		"""Init."""
		self.window = window
		self.overlap = overlap
		self.fs = fs
		self.edges = edges

	def fit(self, X, y=None):
		"""Fit.

		Do nothing. For compatibility purpose.

		Parameters
		----------
		X : ndarray, shape (n_trials, n_channels, n_samples)
			ndarray of trials.
		y : ndarray shape (n_trials,)
			labels corresponding to each trial, not used.

		Returns
		-------
		self : CospCovariances instance
			The CospCovariances instance.
		"""
		return self

	def transform(self, X):
		"""Estimate the cospectral covariance matrices.

		Parameters
		----------
		X : ndarray, shape (n_trials, n_channels, n_samples)
			ndarray of trials.

		Returns
		-------
		covmats : ndarray, shape (n_trials, n_channels, n_channels, n_freq)
			ndarray of covariance matrices for each trials and for each
			frequency bin.
		"""
		Nt, Ne, _ = X.shape
		out = []

		for i in range(Nt):
			S = spectral_edge_frequency(X[i], window=self.window, fs=self.fs,
						   overlap=self.overlap, edges=self.edges)
			out.append(S)

		return np.array(out)


from numpy import unwrap, angle
from scipy.signal import hilbert
from sklearn.base import BaseEstimator, TransformerMixin

class PLV(BaseEstimator, TransformerMixin):
	"""
	Class to extracts Phase Locking Value (PLV) between pairs of channels.
	"""
	def __init__(self, order=100):
		"""Init."""
		self.order = order
		pass

	def fit(self, X, y=None):
		"""
		Fit, do nothing
		"""
		return self

	def plv(self, X):
		n_ch, time = X.shape
		n_pairs = n_ch*(n_ch-1)/2
		# initiate matrices
		phases = np.zeros((n_ch, time))
		delta_phase_pairwise = np.zeros((n_pairs, time))
		plv = np.zeros((n_pairs,))

		# extract phases for each channel
		for c in range(n_ch):
			phases[c, :] = unwrap(angle(hilbert(X[c, :])))

		# compute phase differences
		k = 0
		for i in range(n_ch):
			for j in range(i+1, n_ch):
				delta_phase_pairwise[k, :] = phases[i, :]-phases[j, :]
				k += 1

		# compute PLV
		for k in range(n_pairs):
			plv[k] = np.abs(np.sum(np.exp(1j*delta_phase_pairwise[k, :]))/time)

		return plv

	def transform(self, X):

		out = []
		for x in X:
			tmp = self.plv(x)
			out.append(tmp)
		return np.array(out)




############## NEW FEATURES

############## NON-LINEAR 

import preproc.utils.pymdfa as mfdfa

class MDFA(BaseEstimator, TransformerMixin):
	"""Remove dropped packet."""

	def __init__(self, srate, scstep=8, qstep=4, hp=0.5, lp=12):
		"""Init."""
		self.scstep = scstep
		self.qstep = qstep
		self.hp = hp
		self.lp = lp
		self.srate = srate
		pass

	def fit(self, X, y=None):
		"""
		Fit, do nothing
		"""
		return self

	def get_filt_eeg(self, data_arr):
		n_channels = data_arr.shape[0]
		info = mne.create_info(n_channels, sfreq=self.srate, ch_types = ['eeg']*n_channels)
		raw = mne.io.RawArray(data_arr, info, verbose='ERROR',)

		raw.filter(self.hp, self.lp, verbose='ERROR', fir_design='firwin')
		raw.resample(self.lp*4, npad='auto', verbose='ERROR',)

		return raw.get_data()

	def transform(self, X):
		"""
		Detect and remove dropped.
		"""
		out = []

		scales = np.floor(2.0**np.arange(4, 8, 1.0/self.scstep)).astype('i4') # 8> 10.1
		qs = np.arange(-5,5.01,1.0/self.qstep)

		# R_idx = [0, 1, 2, 3, 8, 9, 10, 14, 15, 16, 17]
		# L_idx = [4, 5, 6, 7, 11, 12, 13, 18, 19, 20, 21]

		# A_idx = [0, 1, 4, 5, 14, 15, 18, 19]
		# P_idx = [2, 3, 6, 7, 16, 17, 20, 21]


		R_idx = [0, 1, 2, 3, 8, 9, 12, 13, 14, 15]
		L_idx = [4, 5, 6, 7, 10, 11, 16, 17, 18, 19]

		A_idx = [0, 1, 4, 5, 12, 13, 16, 17]
		P_idx = [2, 3, 6, 7, 14, 15, 18, 19]

		for cur, x in enumerate(X):
			x = self.get_filt_eeg(x)
			# print("x.shape:", x.shape)
			hqs = [ ]
			Dqs = [ ]
			for c, ch in enumerate(x):
				# print('####:', c)
				# print(ch[:3])
				ch*=1000
				RW = mfdfa.rwalk(ch)
				RMS = mfdfa.fastRMS(RW,scales)
				Fq = mfdfa.compFq(RMS,qs)
				Hq = np.zeros(len(qs),'f8')
				for qi,q in enumerate(qs):
					C = np.polynomial.polynomial.polyfit(np.log2(scales),np.log2(Fq[:,qi]),1)
					Hq[qi] = C[0]
					if abs(q - int(q)) > 0.1: continue
					# loglog(scales,2**np.polynomial.polynomial.polyval(C,log2(scales)),lw=0.5,label='q=%d [H=%0.2f]'%(q,Hq[qi]))
				tq = Hq*qs - 1
				hq = np.diff(tq)/(qs[1]-qs[0])
				Dq = (qs[:-1]*hq) - tq[:-1]

				hqs.append(hq)
				Dqs.append(Dq)

			hqs = np.vstack(hqs)
			Dqs = np.vstack(Dqs)

			hq_mean = np.mean(hqs)
			Dq_mean = np.mean(Dqs)

			hqs_lat_diff = np.mean(hqs[R_idx]) - np.mean(hqs[L_idx])
			hqs_lat_asym = np.abs(hqs_lat_diff)

			hqs_ap_diff = np.mean(hqs[A_idx]) - np.mean(hqs[P_idx])
			hqs_ap_asym = np.abs(hqs_ap_diff)

			hq_var = np.var(hqs)
			Dq_var = np.var(Dqs)

			tmp = np.c_[hq_mean, Dq_mean, hq_var, Dq_var, hqs_lat_diff, hqs_lat_asym, hqs_ap_diff, hqs_ap_asym]

			out.append(tmp)
			
		return np.array(out)


from pyrqa.settings import Settings as pyrqa_Settings
from pyrqa.time_series import TimeSeries
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from datetime import datetime
from sklearn.preprocessing import StandardScaler, RobustScaler
from pyrqa.opencl import OpenCL

class rqa(BaseEstimator, TransformerMixin):
	"""Remove dropped packet."""

	def __init__(self, hp=0.5, lp=4, tau=1, emb_dim=3, sfreq=256):
		"""Init."""
		from pyrqa.opencl import OpenCL
		self.hp = hp
		self.lp = lp
		self.tau = tau 
		self.emb_dim = emb_dim
		self.sfreq = sfreq
		pass

	def fit(self, X, y=None):
		"""
		Fit, do nothing
		"""
		return self

	def get_filt_eeg(self, data_arr):
		with suppress_stdout():
			n_channels = data_arr.shape[0]
			info = mne.create_info(n_channels, 
				sfreq=self.sfreq, 
				ch_types = ['eeg']*n_channels, 
				verbose=False)
			raw = mne.io.RawArray(data_arr, info)

			raw.filter(self.hp, self.lp, verbose='ERROR', fir_design='firwin')
			raw.resample(self.lp*4, npad='auto')

		return raw.get_data()

	def transform(self, X):
		"""
		Detect and remove dropped.
		"""
		opencl = OpenCL(platform_id=0, device_ids=(0,))
		out = []

		R_idx = [0, 1, 2, 3, 8, 9, 10, 14, 15, 16, 17]
		L_idx = [4, 5, 6, 7, 11, 12, 13, 18, 19, 20, 21]

		A_idx = [0, 1, 4, 5, 14, 15, 18, 19]
		P_idx = [2, 3, 6, 7, 16, 17, 20, 21]

		# startTime = datetime.now()

		for cur, x in enumerate(X):
			# RR, DET, LAM, L_max, L_entr, L_mean, and TT

			x = self.get_filt_eeg(x)

			RR = [ ]
			DET = [ ]
			LAM = [ ]
			L_max = [ ]
			L_entr = [ ]
			L_mean = [ ]
			TT = [ ]

			scaler = StandardScaler()
			scaler.fit(x)
			x = scaler.transform(x)
			
			for ch in x:
				# ch*=1000
				ch = TimeSeries(ch,
						 embedding_dimension=self.emb_dim,
						 time_delay=self.tau)
				settings = pyrqa_Settings(ch,
						neighbourhood=FixedRadius(0.5),  #1.0
						similarity_measure=EuclideanMetric,
						theiler_corrector=1)
				computation = RQAComputation.create(settings, opencl=opencl, verbose=False)
				result = computation.run()
				
				result.min_diagonal_line_length=2
				result.min_vertical_line_length=2
				result.min_white_vertical_line_length=2

				curr_RR = result.recurrence_rate
				curr_DET = result.determinism
				curr_LAM = result.laminarity
				curr_L_max = result.longest_diagonal_line # L_max
				curr_L_entr = result.entropy_diagonal_lines # L_entr
				curr_L_mean = result.average_diagonal_line #L_mean
				curr_TT = result.trapping_time #TT

				RR.append(curr_RR)
				DET.append(curr_DET)
				LAM.append(curr_LAM)
				L_max.append(curr_L_max)
				L_entr.append(curr_L_entr)
				L_mean.append(curr_L_mean)
				TT.append(curr_TT)

			RR = np.vstack(RR)
			DET = np.vstack(DET)
			LAM = np.vstack(LAM)
			L_max = np.vstack(L_max)
			L_entr = np.vstack(L_entr)
			L_mean = np.vstack(L_mean)
			TT = np.vstack(TT)

			RR_av = np.mean(RR)
			RR_std = np.std(RR)
			DET_av = np.mean(DET)
			DET_std = np.std(DET)
			LAM_av = np.mean(LAM)
			LAM_std = np.std(LAM)
			L_max_av = np.mean(L_max)
			L_entr_av = np.mean(L_entr)
			L_mean_av = np.mean(L_mean)
			TT_av = np.mean(TT)

			tmp = np.c_[RR_av, RR_std, DET_av, DET_std, LAM_av, LAM_std, L_max_av, L_entr_av, L_mean_av, TT_av]

			out.append(tmp)
			
		# #Python 2: 
		# print datetime.now() - startTime 
		# sys.exit()
		return np.array(out)


class rqa_channel(rqa):
	"""Remove dropped packet."""

	def transform(self, X):
		"""
		Detect and remove dropped.
		"""
		opencl = OpenCL(platform_id=0, device_ids=(0,))
		out = np.zeros([X.shape[0], X.shape[1], 7])

		for cur, x in enumerate(X):
			# RR, DET, LAM, L_max, L_entr, L_mean, and TT
			x = self.get_filt_eeg(x)

			rr = np.zeros(len(x))
			det = np.zeros(len(x))
			lam = np.zeros(len(x))
			l_max = np.zeros(len(x))
			l_entr = np.zeros(len(x))
			l_mean = np.zeros(len(x))
			tt = np.zeros(len(x))

			scaler = StandardScaler()
			scaler.fit(x)
			x = scaler.transform(x)

			for idx,ch in enumerate(x):
				# ch*=1000
				ch = TimeSeries(ch,
						 embedding_dimension=self.emb_dim,
						 time_delay=self.tau)
				settings = pyrqa_Settings(ch,
						neighbourhood=FixedRadius(0.5),  #1.0
						similarity_measure=EuclideanMetric,
						theiler_corrector=1)
				computation = RQAComputation.create(settings, opencl=opencl, verbose=False)
				result = computation.run()
				
				result.min_diagonal_line_length=2
				result.min_vertical_line_length=2
				result.min_white_vertical_line_length=2

				rr[idx] = result.recurrence_rate
				det[idx] = result.determinism
				lam[idx] = result.laminarity
				l_max[idx] = result.longest_diagonal_line # L_max
				l_entr[idx] = result.entropy_diagonal_lines # L_entr
				l_mean[idx] = result.average_diagonal_line #L_mean
				tt[idx] = result.trapping_time #TT

			out[cur] = np.c_[rr, det, lam, l_max, l_entr, l_mean, tt]
			
		return out

# import nolds
# class sampEntropy(BaseEstimator, TransformerMixin):
# 	### PROCESSING TIME WAS 6 HOURS PER SUBJECT SO ABORTED

# 	"""Remove dropped packet."""

# 	def __init__(self, emb_dim=2):
# 		"""Init."""
# 		self.emb_dim = emb_dim
# 		pass

# 	def fit(self, X, y=None):
# 		"""
# 		Fit, do nothing
# 		"""
# 		return self

# 	def transform(self, X):
# 		"""
# 		Detect and remove dropped.
# 		"""
# 		out = []

# 		R_idx = [0, 1, 2, 3, 8, 9, 10, 14, 15, 16, 17]
# 		L_idx = [4, 5, 6, 7, 11, 12, 13, 18, 19, 20, 21]

# 		A_idx = [0, 1, 4, 5, 14, 15, 18, 19]
# 		P_idx = [2, 3, 6, 7, 16, 17, 20, 21]

# 		for cur, x in enumerate(X):
# 			ses = [ ]
# 			for ch in x:
# 				ch*=1000
# 				se = nolds.sampen(ch, self.emb_dim)
# 				ses.append(se)

# 			ses = np.vstack(ses)

# 			se_mean = np.mean(ses)
# 			se_var = np.var(ses)

# 			import IPython; IPython.embed()
# 			se_lat_diff = np.mean(ses[R_idx]) - np.mean(ses[L_idx])
# 			se_lat_asym = np.abs(se_lat_diff)

# 			se_ap_diff = np.mean(ses[A_idx]) - np.mean(ses[P_idx])
# 			se_ap_asym = np.abs(se_ap_diff)


# 			tmp = np.c_[se_mean, se_var, se_lat_asym, se_ap_asym]

# 			out.append(tmp)
			
# 		return np.array(out)

# import rpy2.robjects as ro
# import rpy2.robjects.numpy2ri
# from rpy2.robjects.packages import importr
# class MSMVSampEn(BaseEstimator, TransformerMixin):
# 	### PROCESSING TIME WAS 6 HOURS PER SUBJECT SO ABORTED
	
# 	"""Remove dropped packet."""

# 	def __init__(self, emb_dim=2):
# 		"""Init."""
# 		self.emb_dim = emb_dim
# 		pass

# 	def fit(self, X, y=None):
# 		"""
# 		Fit, do nothing
# 		"""
# 		return self

# 	def transform(self, X):
# 		"""
# 		Detect and remove dropped.
# 		"""

# 		rpy2.robjects.numpy2ri.activate()
# 		MSMVSampEn = importr('MSMVSampEn')

# 		r = 0.5
# 		eps = 10

# 		out = []

# 		R_idx = [0, 1, 2, 3, 8, 9, 10, 14, 15, 16, 17]
# 		L_idx = [4, 5, 6, 7, 11, 12, 13, 18, 19, 20, 21]

# 		A_idx = [0, 1, 4, 5, 14, 15, 18, 19]
# 		P_idx = [2, 3, 6, 7, 16, 17, 20, 21]

# 		for cur, x in enumerate(X):
# 				nr, nc = x.shape
# 				xr = ro.r.matrix(x, nrow=nr, ncol=nc)
# 				M = ro.IntVector([2]*nr) 
# 				tau = ro.IntVector([2]*nr) 

# 				E = MSMVSampEn.MSMVSampEn(xr, M, tau, r, eps)

# 				tmp = np.c_[E]

# 				out.append(tmp)
			
# 		return np.array(out)


############## ARF FEATURES
class LineLength(BaseEstimator, TransformerMixin):
	"""Remove dropped packet."""

	def __init__(self):
		"""Init."""
		pass

	def fit(self, X, y=None):
		"""
		Fit, do nothing
		"""
		return self

	def transform(self, X):
		"""
		Detect and remove dropped.
		"""
		out = []
		for x in X:
			ll = np.abs(np.diff(x))
			tmp = np.c_[ll]
			#tmp = np.c_[m, sd, ku, sk, p90, p10]

			out.append(tmp)
		return np.array(out)


class SimpleStats(BaseEstimator, TransformerMixin):
	"""Remove dropped packet."""

	def __init__(self):
		"""Init."""
		pass

	def fit(self, X, y=None):
		"""
		Fit, do nothing
		"""
		return self

	def transform(self, X):
		"""
		Detect and remove dropped.
		"""
		out = []
		for x in X:
			mn = np.mean(np.abs(x), 1)
			ll = np.sum(np.abs(np.diff(x)), 1)
			sd = np.std(x, 1)
			ku = sp.stats.kurtosis(x, 1)
			sk = sp.stats.skew(x, 1)
			p90 = np.percentile(np.abs(x), 90, axis=1)

			tmp = np.c_[mn, ll, sd, ku, sk, p90]
			out.append(tmp)
		return np.array(out)





############## ADDED FEATURES 12/20/2020

from fooof import FOOOF

def get_FOOOF(spectrum, fvec, freq_range):
	fm = FOOOF(verbose=False)
	fm.fit(fvec, spectrum, freq_range)
	aperiodic_exp = fm.aperiodic_params_[1]
	# aperiodic_fit_spectrum = fm._ap_fit
	# alpha = get_band_peak_fm(fm, [1, 14], select_highest=True)
	return aperiodic_exp #, aperiodic_fit_spectrum, alpha


class AperiodicFeatures(BaseEstimator, TransformerMixin):

	"""Relative power
	"""

	def __init__(self, window=256, overlap=0.5, fs=200, averageAcrossChans=False, freq_range= [1, 40],
				): 
		"""Init."""
		self.window = window
		self.overlap = overlap
		self.fs = fs
		self.averageAcrossChans = averageAcrossChans
		self.freq_range = freq_range

	def fit(self, X, y=None):
		"""Fit.

		Do nothing. For compatibility purpose.

		Parameters
		----------
		X : ndarray, shape (n_trials, n_channels, n_samples)
			ndarray of trials.
		y : ndarray shape (n_trials,)
			labels corresponding to each trial, not used.

		Returns
		-------
		self : CospCovariances instance
			The CospCovariances instance.
		"""
		return self

	def transform(self, X):
		"""Estimate the cospectral covariance matrices.

		Parameters
		----------
		X : ndarray, shape (n_trials, n_channels, n_samples)
			ndarray of trials.

		Returns
		-------
		covmats : ndarray, shape (n_trials, n_channels, n_channels, n_freq)
			ndarray of covariance matrices for each trials and for each
			frequency bin.
		"""

		Nt, Ne, _ = X.shape
		out = []


		for i in range(Nt):
			if self.averageAcrossChans:
				pass

			# pre-ictal periods that are ictal are not included for LSTM
			if np.sum(X[i]==0) > X[i].shape[0]:
				out.append(np.ones((X[i].shape[0],))*-1)
				continue

			n_channels = X[i].shape[0]

			info = mne.create_info(n_channels, sfreq=self.fs, ch_types = ['eeg']*n_channels, verbose='CRITICAL')
			raw = mne.io.RawArray(X[i], info, verbose='CRITICAL')

			# raw_copy.set_eeg_reference('average')
			psd, freq = mne.time_frequency.psd_welch(raw, fmin=0.2, fmax=50, n_fft=512, verbose='CRITICAL')

			pre_out = []
			for ch_data in psd:
				S = get_FOOOF(ch_data, freq, self.freq_range)
				pre_out.append(S)
			out.append(pre_out)

		return np.array(out)


import yaml



class AsymmetryFeatures(BasicStats, TransformerMixin):
	""" measures of left-right asymmetry as calculated on basic stats """

	SWEDE_LEFT_RIGHT_INDEX_PAIRS = [(4, 0),
			(5, 1),
			(6, 2),
			(7, 3),
			(11, 8),
			(10, 9),
			(16, 12),
			(17, 13),
			(18, 14),
			(19, 15)]

	UCSF_LEFT_RIGHT_INDEX_PAIRS = [(4, 0),
			(5, 1),
			(6, 2),
			(7, 3),
			(12, 9),
			(10, 11)]

	def __init__(self, abs_flag=False, which_dataset="ucsf"):
		"""Init."""
		self.abs_flag = abs_flag
		if which_dataset == "swedish":
			self.left_right_index_pairs = self.SWEDE_LEFT_RIGHT_INDEX_PAIRS
		elif which_dataset == "ucsf":
			self.left_right_index_pairs = self.UCSF_LEFT_RIGHT_INDEX_PAIRS


	def transform(self, X):
		X = BasicStats.transform(self, X)

		out = []
		if self.abs_flag:
			for x in X:
				tmp = [
						np.abs(x[ldx] - x[rdx]) / (x[ldx] + x[rdx]) \
						for ldx, rdx in self.left_right_index_pairs
					]
				out.append(np.array(tmp))
		else:
			for x in X:
				tmp = [
						(x[ldx] - x[rdx]) / (x[ldx] + x[rdx]) \
						for ldx, rdx in self.left_right_index_pairs
					]
				out.append(np.array(tmp))
		return np.array(out)



############## ADDED FEATURES 9/17/2023

class Spectrum(BaseEstimator, TransformerMixin):

	"""Relative power
	"""

	def __init__(self, window=256, overlap=0.5, fs=200, freq_range= [1, 40],
				): 
		"""Init."""
		self.window = window
		self.overlap = overlap
		self.fs = fs
		self.averageAcrossChans = averageAcrossChans
		self.freq_range = freq_range

	def fit(self, X, y=None):
		"""Fit.

		Do nothing. For compatibility purposes.

		Parameters
		----------
		X : ndarray, shape (n_trials, n_channels, n_samples)
			ndarray of trials.
		y : ndarray shape (n_trials,)
			labels corresponding to each trial, not used.

		Returns
		-------
		self : TBD
		"""
		return self

	def transform(self, X):
		"""Estimate the power spectrum of EEG signal.

		Parameters
		----------
		X : ndarray, shape (n_trials, n_channels, n_samples)
			ndarray of trials.

		Returns
		-------
		out : ndarray, shape (n_trials, n_channels, n_channels, n_freq)
			ndarray of PSD for each trials 
		"""

		Nt, Ne, _ = X.shape
		out = []


		for i in range(Nt):

			n_channels = X[i].shape[0]

			info = mne.create_info(n_channels, sfreq=self.fs, ch_types = ['eeg']*n_channels, verbose='CRITICAL')
			raw = mne.io.RawArray(X[i], info, verbose='CRITICAL')

			# raw_copy.set_eeg_reference('average')
			psd, freq = mne.time_frequency.psd_welch(raw, fmin=0.2, fmax=50, n_fft=512, verbose='CRITICAL')


		return np.array(psd)

