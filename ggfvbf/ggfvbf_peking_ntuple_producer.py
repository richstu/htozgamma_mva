#!/usr/bin/env python3
import uproot
import numpy as np
import os
import time
import ROOT
import array
import math
import argparse

import sys
sys.path.append('../python_scripts')
import mva_ntuple_producer_tools
import measure_tools

if __name__ == '__main__':
  start_time = time.time()
  np.random.seed(1)

  # Combine signal and background samples into one ntuple AND mix up signal and background event sequence.
  # classID, sampleID, sample_filename_trainrange, sample_filename_fullrange
  class_filenames = [
               [0, 1, 'peking_ntuples/bkgmc.root', 'peking_ntuples/bkgmc_fullrange.root'],
               [1, 2, 'peking_ntuples/signal.root', 'peking_ntuples/signal_fullrange.root'],
               ]
  sampleIDs = [1,2]
  combined_root_name = f'peking_ntuples/peking_combined_sample.root'

  ## Get input ntuples
  #input_ntuples = {}
  #class_list = {}
  #sample_list = {}
  #tree_dict = {'TrainTree':'train_tree', 'ValiTree':'eval_tree', 'TestTree':'test_tree'}
  #for class_id, sample_id, train_filename, baseline_filename in class_filenames:
  #  train_file = uproot.open(train_filename)
  #  baseline_file = uproot.open(baseline_filename)
  #  for tree_name in tree_dict:
  #    train_tree = train_file[tree_name]
  #    baseline_tree = baseline_file[tree_name]
  #    root_branches = train_tree.keys()
  #    # Make keys
  #    if tree_dict[tree_name] not in input_ntuples: 
  #      input_ntuples[tree_dict[tree_name]] = []
  #      input_ntuples[f'{tree_dict[tree_name]}_baseline'] = []
  #      class_list[tree_dict[tree_name]] = []
  #      class_list[f'{tree_dict[tree_name]}_baseline'] = []
  #      sample_list[tree_dict[tree_name]] = []
  #      sample_list[f'{tree_dict[tree_name]}_baseline'] = []
  #    # Fill information
  #    input_ntuples[tree_dict[tree_name]].append(train_tree.arrays(root_branches, '1', library='np'))
  #    input_ntuples[f'{tree_dict[tree_name]}_baseline'].append(baseline_tree.arrays(root_branches, '1', library='np'))
  #    class_list[tree_dict[tree_name]].append(class_id)
  #    class_list[f'{tree_dict[tree_name]}_baseline'].append(class_id)
  #    sample_list[tree_dict[tree_name]].append(sample_id)
  #    sample_list[f'{tree_dict[tree_name]}_baseline'].append(sample_id)
  #  train_file.close()
  #  baseline_file.close()
  ## Make output ntuple
  #output_branches = [branch for branch in root_branches]
  #output_ntuple = {}
  #for branch in root_branches:
  #  for tree_name in input_ntuples:
  #    # Make keys
  #    if tree_name not in output_ntuple: output_ntuple[tree_name] = {}
  #    # Add ntuples into output
  #    concat_list = []
  #    for ntuple in input_ntuples[tree_name]:
  #      concat_list.append(np.squeeze(ntuple[branch]))
  #    output_ntuple[tree_name][branch] = np.float32(np.concatenate(concat_list,0))
  #    # Add classID into output
  #    concat_list = []
  #    for iSample, class_id in enumerate(class_list[tree_name]):
  #      concat_list.append(np.array([class_id]*len(np.squeeze(input_ntuples[tree_name][iSample][root_branches[0]]))))
  #    output_ntuple[tree_name]['classID'] = np.int32(np.concatenate(concat_list,0))
  #    # Add sampleID into output
  #    concat_list = []
  #    for iSample, sample_id in enumerate(sample_list[tree_name]):
  #      concat_list.append(np.array([sample_id]*len(np.squeeze(input_ntuples[tree_name][iSample][root_branches[0]]))))
  #    output_ntuple[tree_name]['sampleID'] = np.int32(np.concatenate(concat_list,0))
  #output_branches.append('classID')
  #output_branches.append('sampleID')
  ## Randomize event sequence to have a balance between samples
  #for tree_name in output_ntuple:
  #  random_idx = np.random.permutation(len(output_ntuple[tree_name]['classID']))
  #  for branch in output_branches:
  #    output_ntuple[tree_name][branch] = output_ntuple[tree_name][branch][random_idx]
  ## Create combined ntuple
  #with uproot.recreate(combined_root_name) as output_file:
  #  for tree_name in output_ntuple:
  #    output_file[tree_name] = output_ntuple[tree_name]

  # Scale weights
  scaled_root_name = f'peking_ntuples/peking_combined_scaled_sample.root'
  tree_collection = {'train':['train_tree', 'eval_tree', 'test_tree'], 'baseline':['train_tree_baseline', 'eval_tree_baseline', 'test_tree_baseline']}
  # Find nevents for each tree
  # nevents_tree['tree_name'] = [background_number, signal_number]
  nentries_tree = {}
  nevents_tree = {}
  tree_names = [str(key.split(';')[0]) for key in uproot.open(combined_root_name).keys()]
  with uproot.open(combined_root_name) as input_file:
    for tree_name in tree_names:
      ntuple_tree = input_file[tree_name]
      branches = [branch for branch in ntuple_tree.keys()]
      tuples = ntuple_tree.arrays(branches, '1', library='np')
      for branch in branches: tuples[branch] = np.squeeze(tuples[branch])
      signal_mask = tuples['classID'] == 1
      background_mask = tuples['classID'] == 0
      nentries_tree[tree_name] = [len(tuples['classID'][background_mask]), len(tuples['classID'][signal_mask])]
      nevents_tree[tree_name] = [np.sum(tuples['weight'][background_mask]), np.sum(tuples['weight'][signal_mask])]
  # Find scale for train and baseline
  total_events = {'train':{0:16425.52+35694.61, 1:157.71+5.90}, 'baseline':{0:86065.89+173553, 1:172.72+6.47}}
  # scales[tree_name] = [background scale, signal scale]
  scales = {}
  for collection in tree_collection:
    total_entries = 0
    #for tree_name in tree_collection[collection]:
    #  total_entries += nentries_tree[tree_name]
    for tree_name in tree_collection[collection]:
      background_scale = total_events[collection][0]*1./nevents_tree[tree_name][0]
      signal_scale = total_events[collection][1]*1./nevents_tree[tree_name][1]
      scales[tree_name] = [background_scale, signal_scale]
  # Scale up weights
  tuples_dict = {}
  with uproot.open(combined_root_name) as input_file:
    for tree_name in tree_names:
      ntuple_tree = input_file[tree_name]
      branches = [branch for branch in ntuple_tree.keys()]
      tuples_dict[tree_name] = ntuple_tree.arrays(branches, '1', library='np')
      for branch in branches: tuples_dict[tree_name][branch] = np.squeeze(tuples_dict[tree_name][branch])
      background_mask = tuples_dict[tree_name]['classID'] == 0
      signal_mask = tuples_dict[tree_name]['classID'] == 1
      tuples_dict[tree_name]['weight'][background_mask] = tuples_dict[tree_name]['weight'][background_mask]*scales[tree_name][0]
      tuples_dict[tree_name]['weight'][signal_mask] = tuples_dict[tree_name]['weight'][signal_mask]*scales[tree_name][1]
  with uproot.recreate(scaled_root_name) as output_file:
    for tree_name in tree_names:
      output_file[tree_name] = tuples_dict[tree_name]
  print(f'Produced {scaled_root_name}')



  ## Change name of branches
  #name_change = {'weight':'weightXyear', 'Z_cos_theta':'cosTheta', 'lep_cos_theta':'costheta', 'lep_phi':'phi', 'H_relpt':'lly_ptmass', 'l2g_deltaR':'yl_drmin', 'l1g_deltaR':'yl_drmax'}

  elapsed_time = time.time() - start_time
  print(f'Elapsed time: {elapsed_time:.1f} sec')
