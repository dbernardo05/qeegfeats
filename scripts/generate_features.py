# 20180625
import importlib

import pickle

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
# from tqdm.notebook import tqdm
from tqdm import tqdm

from sklearn.pipeline import make_pipeline
hard_coded_paths = yaml.load(open("metadata/hard_coded_paths.yml"),
							 Loader=yaml.FullLoader)
sys.path.append(hard_coded_paths["utils_parent"])
from preproc.utils.eeg_preprocessing import *




class PickleProtocol:
    def __init__(self, level):
        self.previous = pickle.HIGHEST_PROTOCOL
        self.level = level

    def __enter__(self):
        importlib.reload(pickle)
        pickle.HIGHEST_PROTOCOL = self.level

    def __exit__(self, *exc):
        importlib.reload(pickle)
        pickle.HIGHEST_PROTOCOL = self.previous


def pickle_protocol(level):
    return PickleProtocol(level)


def _make_pipeline(feature):
	pipe = []
	for item in feature['preproc']:
		for method, params in item.items():
			pipe.append(_from_yaml_to_func(method, params))

	# create pipeline
	preproc_ = make_pipeline(*pipe)

	postproc_ = None
	# parse pipe function from parameters
	if 'postproc' in feature.keys():
		pipe = []
		for item in yml['postproc']:
			for method, params in item.items():
				pipe.append(_from_yaml_to_func(method, params))

		# create pipeline
		postproc_ = make_pipeline(*pipe)
	return preproc_, postproc_


def _get_fnames(subject, labeled_data_save_dir, data_ext=".h5"):
	base = labeled_data_save_dir + \
		'%s/%s_' % (str(subject).zfill(3), str(subject).zfill(3))

	fnames = sorted(glob(base + "*" + data_ext),
					key = lambda x: int(x.replace(base, "").split("_")[0]))

	if len(fnames) == 0:
		print("data_dir:", base)
		raise Exception("Error. No files found.")
	return fnames


def _load_yaml(fname):
	yml = yaml.load(open(fname), Loader=yaml.FullLoader)
	preproc_arf = yml['preproc_arf'] if 'preproc_arf' in yml else False
	denoise_mode = yml['denoise_mode'] if 'denoise_mode' in yml else True
	bp_mode = yml['bp_mode'] if 'bp_mode' in yml else True
	nf_mode = yml['nf_mode'] if 'nf_mode' in yml else True
	debug_mode = yml['debug_mode'] if 'debug_mode' in yml else False
	labeled_data_save_dir = yml['save_dir'] if 'save_dir' in yml else ""
	srate = yml['srate']

	if debug_mode:
		print("IN DEBUG MODE!!!")

	return yml, preproc_arf, denoise_mode, bp_mode, nf_mode, debug_mode, labeled_data_save_dir, srate


def _from_yaml_to_func(method, params):
	"""Convert yaml to function"""
	prm = dict()
	if params is not None:
		for key, val in params.items():
			try:
				prm[key] = eval(str(val))
			except SyntaxError as e:
				prm[key] = val
	return eval(method)(**prm)

def parse_filepath_for_metadata(full_path_to_file, data_ext=".h5"):
	# 20220620 ~ Dan changed to use attr in dict

	metadata_list = os.path.basename(full_path_to_file).split("_")

	if metadata_list[1][0].isalpha():
		# This is UCSF HIE data which has two char string at second position (index 1), for subsegment
		epoch_idx = int(metadata_list[2])
		subsegment = metadata_list[1]
		annotation = int(metadata_list[3])
		szs_present = int(metadata_list[4])
		augmentation = metadata_list[5].split(data_ext)[0]
	else:
		# This is SWEDE data
		epoch_idx = int(metadata_list[1])
		subsegment = 0
		annotation = int(metadata_list[2])
		szs_present = int(metadata_list[3])
		augmentation = metadata_list[4].split(data_ext)[0]
	augmentation = augmentation.split('-')[-1]
		# if len(metadata_list) > 4:
		# 	augmentation = ("_".join(metadata_list[5:])).split(data_ext)[0] #[-1].split(data_ext)[0]
		# else:
		# 	augmentation = "none"

	basename = os.path.basename(full_path_to_file).split(data_ext)[0]

	return epoch_idx, subsegment, annotation, szs_present, augmentation, basename


def process_single_h5(preproc_, fname, data_ext, srate, 
					   denoise_mode=False, bp_mode=False, debug_mode=False, 
					   nf_mode=True, montage='ucsf_nicu_montage'):
	
	epoch_idx, subsegment, y, szs_present, augmentation, basename = \
		parse_filepath_for_metadata(fname)

	with h5py.File(fname, 'r') as hf:
		eeg = hf[basename][:]

	if denoise_mode:
		eeg = denoise(eeg)

	if bp_mode:
		eeg = preproc_eeg(eeg, srate, montage, nf_mode=nf_mode)

	# print('\t\teeg shape:', eeg.shape)
	X = preproc_.fit_transform(np.array([eeg]))
	# print('\t\tX shape:', X.shape)

	if len(X) == 1:
		X = X[0]

	# Allow 1% tolerance of NaN; put downstream in future
	if np.sum(np.isnan(X)) < X.size*0.01:
		X[np.isnan(X)] = 0
	else:
		print('Warning: >1% NaN for file:', fname)
		print(np.sum(np.isnan(X)))

	valid = np.sum(np.isnan(X)) == 0

	if np.all(eeg == 0):  # if data is all zeros, it is invalid!
		valid = False
		X[:] = np.nan

	return X, y, valid, epoch_idx, subsegment, basename, augmentation


if __name__ == '__main__':

	# parse command line args
	parser = OptionParser()

	parser.add_option("-s", "--subject",
					  dest="subject", default=-1,
					  help="The subject")
	parser.add_option("-c", "--config",
					  dest="config", default="config.yml",
					  help="The config file")
	parser.add_option("-n", "--njobs",
					  dest="njobs", default=1,
					  help="the number of jobs")
	parser.add_option("-i",
					  dest="input_dir")
	parser.add_option("-o",
					  dest="output_dir")
	parser.add_option("-f",
					  dest="readable_feats", default="metadata/human_readable_feat_names_ucsf.yml")
	parser.add_option("-a", "--augmentation",
					  dest="augmentation", default=0)
	(options, args) = parser.parse_args()
	augmentation = int(options.augmentation)
	subject = int(options.subject)
	njobs = int(options.njobs)
	epoch_len = 60

	# load yaml file
	yml, preproc_arf, denoise_mode, bp_mode, nf_mode, debug_mode, _, srate = _load_yaml(
		options.config)
	
	# imports
	for pkg, functions in yml['imports'].items():
		if 'utils' in pkg: # already loaded
			pkg = 'preproc.' + pkg # temp fix for module loading bug, maybe linux vs mac issue?
		stri = 'from ' + pkg + ' import ' + ','.join(functions)
		exec(stri)

	# set up folders for npz files to land
	output_dirs = []
	human_readable_feature_names = \
		yaml.load(open(options.readable_feats), 
			Loader=yaml.FullLoader)
	
	for output in yml['feature_output']:  # create forlder if it does not exist
		output_dir = os.path.join(options.output_dir, output) #"./features/%s" % output
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		output_dirs.append(output_dir)
		

	# only pull data that exists
	subjs = [x.split("/")[-1] for x in glob(options.input_dir + "*") if \
			 os.path.isdir(x)] 

	for subject in tqdm.tqdm(np.sort(subjs)[0:]):
		# allow for multiple feature sets to be generated with one execution of this script
		for output_dir, feature in zip(output_dirs, yml['features']):
			wins_per_epoch = int(epoch_len / feature['preproc'][0]['Windower']['window'])
			preproc_, postproc_ = _make_pipeline(feature)

			h5_fnames = _get_fnames(subject, options.input_dir, data_ext=".h5")

			if not augmentation:
				h5_fnames = [fn for fn in h5_fnames if 'none' in fn]
				# h5_fnames = [fn for fn in h5_fnames if 'none' in fn or 'aug-horizflip' in fn]

			if 0:
				# Debugging
				Xs, ys, valids, epoch_indices, subsegments, fnames, augmentations = [], [], [], [], [], [], []
				for ii, fname in enumerate(h5_fnames):
					res = process_single_h5(preproc_, fname=fname, data_ext=".h5", srate=srate, 
								debug_mode=debug_mode, bp_mode=bp_mode, 
								denoise_mode=denoise_mode, nf_mode=nf_mode, montage='ucsf_nicu_montage')
					Xs_, ys_, valids_, epoch_indices_, subsegments_, fnames_, augmentations_ = res
					print(Xs_.shape)
					Xs.append(Xs_)
					ys.append(ys_)
					valids.append(valids_)
					epoch_indices.append(epoch_indices_)
					subsegments.append(subsegments_)
					fnames.append(fnames_)
					augmentations.append(augmentations_)
					print('\t', ii, epoch_indices_, fnames_, Xs_.shape)
					print('\tys_', ys_)
					print(ys_.shape)
					print(Xs_.shape)
					sys.exit()

			else:
				res = Parallel(n_jobs=njobs, verbose=0)(
						delayed(process_single_h5)(preproc_=preproc_, fname=fname, data_ext=".h5", srate=srate, 
							debug_mode=debug_mode, bp_mode=bp_mode, 
							denoise_mode=denoise_mode, nf_mode=nf_mode, montage='ucsf_nicu_montage')
							for ii, fname in enumerate(h5_fnames))

				Xs, ys, valids, epoch_indices, subsegments, fnames, augmentations = zip(*res)


			Xs = np.array(Xs)
			ys = np.array(ys)
			valids = np.array(valids)
			epoch_indices = np.array(epoch_indices)
			subsegments = np.array(subsegments)
			fnames = np.array(fnames)
			augmentations = np.array(augmentations)


			assert len(fnames) == len(Xs)

			if postproc_ is not None:
				print("\npost process training data")
				Xs = postproc_.fit_transform(Xs[valid], y[valid])
				out_shape = list(Xs.shape)
				out_shape[0] = len(valid)
				Xs_final = np.ones(tuple(out_shape)) * np.nan
				Xs_final[valid] = Xs
			else:
				Xs_final = Xs

			if len(Xs_final.shape) == 4: # multiple windows in a single epoch
				Xs_flattened = Xs_final.reshape(
					[Xs_final.shape[0] * Xs_final.shape[1], -1])
			elif len(Xs_final.shape) == 3:
				Xs_flattened = Xs_final.reshape([Xs_final.shape[0], -1])
			else:
				print("number of dims wrong: %d" % len(Xs_final.shape))
				print("double check epoch len and window length in config")
				import IPython; IPython.embed()

			feat_set = output_dir.split("/")[-1].split("_")[0]
			if feat_set in human_readable_feature_names:
				feature_column_names = human_readable_feature_names[feat_set]
			else:
				feature_column_names = ["%s_feat%d" % (feat_set, i) \
										for i in range(0, Xs_flattened.shape[-1])]

			if 0:
				# Debugging
				print(feat_set)
				print(Xs.shape)
				print('#######')
				print(len(feature_column_names))
				print(len(Xs_flattened))
				print(Xs_flattened.shape)
				print('wins_per_epoch:', wins_per_epoch)
				print('ys:', ys)
				sys.stdout.flush()

			df = pd.DataFrame(data=Xs_flattened, 
				columns=feature_column_names)
			df["y"] = ys.repeat(wins_per_epoch) 
			df["valid"] = valids.repeat(wins_per_epoch)
			df["subsegment"] = subsegments.repeat(wins_per_epoch)
			df["fname"] = fnames.repeat(wins_per_epoch)
			df["augmentation"] = augmentations.repeat(wins_per_epoch)
			df["idx"] = epoch_indices.repeat(wins_per_epoch)

			dfs_split_out_by_augmentation = [df[df["augmentation"] == aug] \
				for aug in sorted(np.unique(augmentations))]
			df_reordered = pd.concat(dfs_split_out_by_augmentation).reset_index()
			trust = ~df_reordered.isnull().any(axis=1)

			print(subject, ": # trusted / # total: ", 
				len(df_reordered[trust]), 
				"/", 
				len(df_reordered))
			save_name = "%s/%s.h5" % (output_dir, str(subject).zfill(3))
			with pickle_protocol(4):
				df_reordered.to_hdf(save_name, "data")
			sys.stdout.flush()
			# clear memory
			del res
			del df
			del Xs_flattened
			del Xs_final
			del Xs

		gc.collect()

