import pickle as pkl

import mne
import numpy as np
import pandas as pd
import h5py
import os
import re
import sys
import yaml
import multiprocessing
import gc

#from scipy.io import loadmat
from glob import glob

from progressbar import Bar, ETA, Percentage, ProgressBar
from joblib import Parallel, delayed
from optparse import OptionParser
import tqdm

from sklearn.pipeline import make_pipeline

# from metadata.montage_channel_configurations import *


SWEDE_CH_LEGEND = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
				   'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 
				   'Cz', 'Pz']

# UCSF_CH_LEGEND = ['C3', 'C4', 'O1', 'O2', 'Cz', 'Fp1', 'Fp2', 'T7', 'T8']
UCSF_CH_LEGEND = ['C3', 'C4', 'O1', 'O2', 'Cz', 'Fp1', 'Fp2', 'T3', 'T4']

SWEDE_NICU_TCP_MONTAGE = [
	['fp2', 'f8'],
	['f8', 't4'],
	['t4', 't6'],
	['t6', 'o2'],

	['fp1', 'f7'],
	['f7', 't3'],
	['t3', 't5'],
	['t5', 'o1'],

	['t4', 'c4'],  # 8
	['c4', 'cz'],
	['cz', 'c3'],  # original montage w/out symmetry
	['c3', 't3'],
	# ['c3', 'cz'], # modifying to maintain central symmetry
	# ['t3', 'c3'],

	['fp2', 'f4'],  # 12
	['f4', 'c4'],
	['c4', 'p4'],
	['p4', 'o2'],

	['fp1', 'f3'],
	['f3', 'c3'],
	['c3', 'p3'],
	['p3', 'o1'],
]

NICU_MONTAGE = [
	['fp2', 't4'],
	['t4', 'o2'],

	['fp1', 't3'],
	['t3', 'o1'],

	['t4', 'c4'],
	['c4', 'cz'],
	['cz', 'c3'],
	['c3', 't3'],

	['fp2', 'c4'],
	['c4', 'o2'],

	['fp1', 'c3'],
	['c3', 'o1'],

	['t3', 'fp1'],
	['fp1', 'fp2'],
	['fp2', 't4'],

	['t3', 'o1'],
	['o1', 'o2'],
	['o2', 't4'],

]

UCSF_NICU_MONTAGE = [
	['fp1', 'c3'],  # 0
	['c3', 'o1'], 	# 1
	['fp1', 't3'],	# 2
	['t3', 'o1'],	# 3
	['fp2', 'c4'],	# 4
	['c4', 'o2'],	# 5
	['fp2', 't4'],	# 6
	['t4', 'o2'],	# 7

	['fp2', 'fp1'],	# 8
	['t4', 'c4'],	# 9
	['c4', 'cz'],	# 10
	['cz', 'c3'],	# 11
	['c3', 't3'],	# 12
	['o2', 'o1'],	# 13
]

UCLA_TCP_MONTAGE = [
	['fp2', 'f8'],
	['f8', 't4'],
	['t4', 't6'],
	['t6', 'o2'],

	['fp1', 'f7'],
	['f7', 't3'],
	['t3', 't5'],
	['t5', 'o1'],

	['a2', 't4'],
	['t4', 'c4'],
	['c4', 'cz'],
	['cz', 'c3'],
	['c3', 't3'],
	['t3', 'a1'],

	['fp2', 'f4'],
	['f4', 'c4'],
	['c4', 'p4'],
	['p4', 'o2'],

	['fp1', 'f3'],
	['f3', 'c3'],
	['c3', 'p3'],
	['p3', 'o1'],
]

def zero_runs(a):
	# Create an array that is 1 where a is 0, and pad each end with an extra 0.
	iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
	absdiff = np.abs(np.diff(iszero))
	# Runs start and end where absdiff is 1.
	ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
	return ranges


def denoise(data):
	data_mod = np.zeros_like(data)
	len_data = data.shape[1]

	for c, ch in enumerate(data):
		ch_mod = np.copy(ch)
		ch_mod[ch_mod > 0.001] = 0
		ch_mod[ch_mod < -0.001] = 0

		z = zero_runs(ch_mod)

		z_cx = np.where(np.diff(np.signbit(ch)))[0]  # zero crossings

		for locs in z:
			st, nd = locs
			if st == 0:
				st_merge = 0
			elif np.any(z_cx[z_cx < st]):
				st_merge = z_cx[z_cx < st][-1] - 1
			else:
				curr_val = np.abs(ch[st])
				new_val = np.abs(ch[st-1])
				nn = 0
				while new_val < curr_val:
					nn += 1
					curr_val = np.abs(ch[st-nn])
					new_val = np.abs(ch[st-nn-1])
					if st-nn-1 == 0:
						break
				st_merge = st - nn

			try:
				nd_merge = z_cx[z_cx > st][0] + 1
			except:
				nd_merge = nd

			ch_mod[st_merge: nd_merge] = 0

		# Get noise-free data
		z = zero_runs(ch_mod)
		if z.shape[0] == 0:
			#plt.plot(ch_mod[view_rng]*50 + c*0.1, alpha=0.8)
			continue
		elif z.shape[0] == 1:
			nz = np.array([[0, z[0, 0]], [z[0, 1], len_data]])
			nz_durs = nz[:, 1] - nz[:, 0]
		else:
			nz = np.zeros_like(z)
			nz[:, 0] = z[:, 1]
			if nz.shape[0] == 0:
				print('err')
				print(z)
				print(nz)
				sys.exit()
			nz[:-1, 1] = z[1:, 0]
			nz[-1, 1] = len_data

			if z[0, 0] != 0:
				nz[0, 0] == 0

			nz_durs = nz[:, 1] - nz[:, 0]

		if np.array_equal(z, np.array([[0, len_data]])):
			continue

		for locs in z:
			st, nd = locs
			dur = nd - st
			nz_idx = np.logical_and((nz_durs >= dur + 2), nz_durs > 2)
			nz_durs_sel = nz_durs[nz_idx]
			if nz_durs_sel.shape[0] == 0:
				abs_idx = np.argmax(nz_durs)
				#nz_idx[abs_idx] = True
				mult = np.ceil(dur/np.amax(nz_durs)) + 1

			else:
				mult = None
				idx = np.argmax(nz_durs_sel)
				abs_idx = np.where(nz_idx)[0][idx]

			nz_st, nz_nd = nz[abs_idx]

			if mult:
				rep = np.repeat(ch_mod[nz_st:nz_nd], mult)
				ch_mod[st: nd] = rep[:nd-st]

			else:
				if (nz_nd - dur - 1) <= nz_st:
					print('Error w/inserting noise free data.')
					print(mult)
					print('nz_dur:', nz_durs[abs_idx])
					print(nz_st, nz_nd, dur)
					print(st, nd)
					sys.exit()
				else:

					nz_st = np.random.randint(
						nz_st, nz_nd - dur - 1)  # Select start

				ch_mod[st: nd] = ch_mod[nz_st:nz_st + dur]

			data[c, :] = ch_mod
	return data


def get_filt_eeg(data_arr, numeric_montage, channel_names, srate):
	data_list = []
	len_data = data_arr[0].shape[0]
	num_chans = len(numeric_montage)

	for ch, chan in enumerate(numeric_montage):
		curr_ch = chan[0]
		if curr_ch is not None:
			data_list.append(data_arr[curr_ch])
		else:
			data_list.append(np.zeros((len_data,)))

	raw_N300_N180 = np.copy(data_list)
	info = mne.create_info(ch_names=[str(x) for x in range(
		len(data_list))], sfreq=srate, ch_types=num_chans*['eeg'], 
		verbose='ERROR')
	raw = mne.io.RawArray(raw_N300_N180, info, verbose='ERROR')
	raw = raw.notch_filter(
		60, phase='zero', fir_design='firwin', verbose='ERROR')
	try:
		raw = raw.filter(0.1, None, fir_design='firwin', verbose='ERROR')
	except IndexError as e:
		import IPython; IPython.embed()
	return raw.get_data()


def get_montage(sel_montage):
	if sel_montage == 'ucla_tcp_montage':
		montage = UCLA_TCP_MONTAGE
	elif sel_montage == 'swede_nicu_tcp_montage':
		montage = SWEDE_NICU_TCP_MONTAGE
	elif sel_montage == 'nicu_montage':
		montage = NICU_MONTAGE
	elif sel_montage == 'ucsf_nicu_montage':
		montage = UCSF_NICU_MONTAGE
	else:
		print("choose correct montage")
		sys.exit()

	channel_names = [''.join((pair[0], '-', pair[1])) for pair in montage]
	return montage, channel_names


def get_ch_loc(ch_legend, ch_oi):
	results = []
	for key, value in enumerate(ch_legend):
		if ch_oi.lower() in value.lower():
			results.append(key)

	if len(results) != 1:
		print('Error in tasks.get_ch_loc()')
		print(results)
		sys.exit()
	else:
		return results[0]


def prepare_montage_channels(data, numeric_montage, montage_channel_names):

	index = np.arange(data.shape[1])
	df_ = pd.DataFrame(
		np.array([data[ch_pair[0]]-data[ch_pair[1]] for ch_pair in numeric_montage]).T,
		index=index,
		columns=montage_channel_names)
	eeg_data = df_.values.T
	return eeg_data


def preproc_eeg(data_arr, srate, montage, nf_mode=True, bp_mode=True):

	# For Swede dataset
	ch_legend = UCSF_CH_LEGEND

	# print('Generating montage...')
	montage, channel_names = get_montage(montage)
	numeric_montage = []
	for ch_pair in montage:
		curr_pair = [get_ch_loc(ch_legend, ch_pair[0]),
					 get_ch_loc(ch_legend, ch_pair[1])]
		numeric_montage.append(curr_pair)

	if nf_mode:
		data_arr = get_filt_eeg(
			data_arr, numeric_montage, channel_names, srate)

	if bp_mode:
		data_arr = prepare_montage_channels(
			data_arr, numeric_montage, channel_names)

	return data_arr

