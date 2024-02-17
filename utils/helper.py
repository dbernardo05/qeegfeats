import os
import pickle as pkl
import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import six
import sys
import xgboost as xgb
import yaml
import seaborn as sns
import mne

from datetime import date

from copy import deepcopy
from glob import glob
from progressbar import Bar, ETA, Percentage, ProgressBar
from joblib import Parallel, delayed, dump
from optparse import OptionParser
from time import time
from scipy import interp
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, KFold
from sklearn.pipeline import make_pipeline

hard_coded_paths = yaml.load(open("metadata/hard_coded_paths.yml"), \
	Loader=yaml.FullLoader)
sys.path.append(hard_coded_paths["utils_parent"])

# Features directory
FEATURES_DIR = os.path.join(hard_coded_paths["base_dir"], "preproc/features")


def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]


def get_hp_data(eeg_data, lf):
    # raw.notch_filter(np.arange(60, 361, 60), picks=picks, filter_length=2048*20, phase='zero')
    eeg_data.filter(lf, None, fir_design='firwin')
    # curr_hp1_data.plot_psd(area_mode='range', tmax=10.0, average=False)
    # eeg_data = eeg_data.get_data()
    return eeg_data


def find_files(search_dir, ext, subdirs):
    matches = []
    ignored = []

    for subdir in subdirs:
        search_subdir = os.path.join(search_dir, subdir)
        print('Curr subdir:', search_subdir)
        for root, dirnames, filenames in os.walk(search_subdir):
            for filename in fnmatch.filter(filenames, '*' + ext):
                fname = os.path.join(root, filename)
                if fname.split('/')[-1][0] == '.':
                    ignored.append(fname)
                else:
                    matches.append(fname)
    print("Ignored:", ignored)
    print("Matches:")
    for m in matches:
        print("\t", m)

    return matches


def get_anns(eeg_data):
    pre_annotations = mne.events_from_annotations(eeg_data)
    anns = []
    for ann in pre_annotations:
        if len(ann) > 0:
            if '#' in ann[2]:
                anns.append([ann[0]*srate, ann[2]])
    return anns


def load_edf(edf_file):
    raw_edf = mne.io.read_raw_edf(edf_file, preload=True, verbose='ERROR')

    orig_rate = int(raw_edf.info['sfreq'])
    # if verbose != "ERROR":
    #     print('\t\tOrig Sample Rate', orig_rate)
    #     print('\t\tChannel Names:', raw_edf.ch_names)

    # fix chans
    new_ch_map = {}
    for ch in raw_edf.ch_names:
        n_ch = ch.replace('EEG ', '').replace('POL ', '').replace(
            '-Ref', '').replace('-REF', '').replace('Z', 'z').replace(' ', '')
        new_ch_map[ch] = n_ch
    raw_edf.rename_channels(new_ch_map)

    # if verbose != "ERROR":
    #     print('\t\tNew Channel Names:', raw_edf.ch_names)

    # For Swede dataset
    ref_chans = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
    			 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']

    raw_edf.pick_channels(ref_chans)

    if len(raw_edf.ch_names) != len(ref_chans):
        print("Missing:")
        print(list(set(ref_chans) - set(raw_edf.ch_names)))
        sys.exit()

    if int(orig_rate) != 256:
        print('Data sample rate not acceptable:', orig_rate)
        sys.exit()
    else:
        hp01 = get_hp_data(raw_edf, 0.1)

    if np.any(np.isnan(hp01.get_data())):
        print('############# ERROR NAN WHEN FILTERING')
        sys.exit()

    # if verbose != "ERROR":
    #     print('\t\tLength data (sec):', (hp01.get_data().shape[1]/orig_rate))

    reref_earelec = False
    if reref_earelec:
        raw_ipsiear_ref, _ = mne.set_eeg_reference(
            hp01, ref_channels=['A1', 'A2'])
        if np.any(np.isnan(raw_ipsiear_ref.get_data())):
            print('############# ERROR NAN WHEN MONTAGING')
            sys.exit()

    return hp01.get_data(), get_anns(raw_edf), orig_rate


def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))


def load_features(subjects, datasets, spqr_data_dir, data_type, split, cv, dtype, art_red_mode=False, big_arf_mode=False, flatten_mode=True):

	with open(os.path.join(hard_coded_paths["base_dir"], 'binaries/bigArf_sco_data.pkl'), 'rb') as fp:
		df = pkl.load(fp, encoding='latin1')

	features = []
	for subject in subjects[split]:
		temp_features = []
		for jj, dataset in enumerate(datasets):
			feat_data_path = os.path.join(FEATURES_DIR, '%s/%s.npz' % (dataset, str(subject).zfill(3)))
			if not os.path.exists(feat_data_path):
				print("%s does not exist, skipping!" % feat_data_path)
				continue
			a = np.load(feat_data_path)
			all_fn = a['fnames']

			sel_fn_idx = np.zeros((len(all_fn),), dtype=bool)
			for f, fn in enumerate(all_fn):
				ba = spqr_data_dir + data_type + '/%s/' % (str(subject).zfill(3))

				if big_arf_mode:
					outlier_sco = df.loc[df['fname'] == ba + fn]['outlier'].values[0]
					if outlier_sco < 3:
						sel_fn_idx[f] = True
				else:
					sel_fn_idx[f] = True


			sel_feats = a['features'][sel_fn_idx]

			# for f, fn in enumerate(sel_feats):
			# 	import IPython; IPython.embed()
			# 	a_t = sel_feats[f].T[:,0,:]
			# 	ab = np.ones((a_t.shape[1] + 4,)) 
					
			temp_features.append(sel_feats)

		# if len(temp_features) > 0:
		if flatten_mode:
			flattened_temp_features = []
			for temp_feature in temp_features:
				if len(temp_feature.shape) > 3:
					flattened_temp_features.append(temp_feature.reshape(temp_feature.shape[0], temp_feature.shape[1], -1))
				else:
					flattened_temp_features.append(temp_feature)
		if len(temp_features) == len(datasets):
			if flatten_mode:
				temp_features = np.concatenate(flattened_temp_features, -1)
			else:
				temp_features = np.concatenate(temp_features, -1)

			fnames = cv['fnames_%s' % dtype]
			fn_idcs = []
			for fn, fname in enumerate(fnames):
				if str(subject).zfill(3) in fname.split('_')[0]:
					fn_idcs.append(fn)
			assert fn_idcs == sorted(fn_idcs)

			if art_red_mode:
				safe_idcs = cv['indices_%s' % dtype][fn_idcs]
				art_idcs = np.logical_not(safe_idcs)
				temp_features[art_idcs, :] = np.nan
			#print 'temp_features.shape:', temp_features.shape


			features.append(temp_features)
	return features


def load_feature_matrix(feat_name, feat_names, spqr_data_dir, load_from_save, analyze_per_window=True, save=True):
	temp_fnames = []
	temp_ys = []
	temp_Xs = []
	temp_sIDs = []
	subj_list = [str(n).zfill(3) for n in range(1, 80)]
	feat_dir = os.path.join('../preproc/features', feat_name)

	feat_save_name = '../preproc/debug/binaries/X_%s.pkl' % (feat_name)
	label_save_name = '../preproc/debug/binaries/y.pkl'

	if load_from_save and os.path.exists(feat_save_name) and os.path.exists(label_save_name):
		X = pd.read_pickle(feat_save_name)
		y = pd.read_pickle(label_save_name)
		return X, y

	for subject in subj_list:
	    base = spqr_data_dir + '/%s/%s_' % (str(subject).zfill(3), str(subject).zfill(3))
	    path_feature = os.path.join(feat_dir, '%s.npz' % (str(subject).zfill(3)))

	    fnames = sorted(glob(base + '*.h5'), key=lambda x: int(x.replace(base, '')[:-7]))
	    fnames_finals = []

	    for f, fname in enumerate(fnames):
	        ba = spqr_data_dir + '/%s/' % (str(subject).zfill(3))
	        fn = fname.replace(ba, '')
	        fnames_finals.append(fn)
	    temp_fnames.extend(fnames_finals)
	    
	    for fname in fnames_finals:
	        temp_ys.extend([int(fname.split("_")[-2])]*3)
	    
	    if os.path.exists(path_feature):
	        x_train = np.load(path_feature)
	        if analyze_per_window:
	        	flat_X_train = x_train['features'].reshape((x_train['features'].shape[0] * x_train['features'].shape[1], 
	                                                    np.prod(x_train['features'].shape[2:])))
	        else:
	        	flat_X_train = x_train['features'].reshape((x_train['features'].shape[0], 
	                                                    np.prod(x_train['features'].shape[1:])))
	            
	        for epoch in flat_X_train:
	            temp_Xs.append(epoch)
	a = pd.DataFrame(np.array(temp_Xs))
	b = a.set_axis(feat_names, axis=1, inplace=False)
	pd.to_pickle(b, feat_save_name)
	pd.to_pickle(temp_ys, label_save_name)
	return b, temp_ys
