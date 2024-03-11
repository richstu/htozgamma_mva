#!/usr/bin/env python3
import uproot
import numpy as np
import os
import time

def combine_ntuples(class_filenames):
  # Get input ntuples
  input_ntuples = []
  class_list = []
  for class_id, filename in class_filenames:
    root_file = uproot.open(filename)
    root_tree = root_file['tree']
    root_branches = root_tree.keys()
    input_ntuples.append(root_tree.arrays(root_branches, '1', library='np'))
    class_list.append(class_id)
    root_file.close()

  # Make output ntuple
  output_branches = [branch for branch in root_branches]
  output_ntuple = {}
  for branch in root_branches:
    # Add ntuples into output
    concat_list = []
    for ntuple in input_ntuples:
      concat_list.append(np.squeeze(ntuple[branch]))
    output_ntuple[branch] = np.float32(np.concatenate(concat_list,0))
    # Add classID into output
    concat_list = []
    for iSample, class_id in enumerate(class_list):
      concat_list.append(np.array([class_id]*len(np.squeeze(input_ntuples[iSample][root_branches[0]]))))
    output_ntuple['classID'] = np.int32(np.concatenate(concat_list,0))
  output_branches.append('classID')

  # Randomize event sequence to have a balance between samples
  random_idx = np.random.permutation(len(output_ntuple['classID']))
  for branch in output_branches:
    output_ntuple[branch] = output_ntuple[branch][random_idx]

  return output_ntuple, output_branches

def split_ntuple(ntuple, branches, uproot_cut, weight_branch_names, output_filename):
  train_fraction = 0.34
  eval_fraction = 0.33
  test_fraction = 1 - train_fraction - eval_fraction

  ## Below doesn't work well. train has 840k, eval has 480k, test has 770k
  ## Split ntuple into trees
  #train_ntuple, eval_ntuple, test_ntuple = {}, {}, {}
  #train_idx = (ntuple['event_number']%10==0) | (ntuple['event_number']%10==3) | (ntuple['event_number']%10==6) | (ntuple['event_number']%10==9)
  #eval_idx =  (ntuple['event_number']%10==1) | (ntuple['event_number']%10==4) | (ntuple['event_number']%10==7)
  #test_idx =  (ntuple['event_number']%10==2) | (ntuple['event_number']%10==5) | (ntuple['event_number']%10==8)
  #for branch in branches:
  #  train_ntuple[branch] = ntuple[branch][train_idx]
  #  eval_ntuple[branch] = ntuple[branch][eval_idx]
  #  test_ntuple[branch] = ntuple[branch][test_idx]

  # Split the ntuples
  stack_ntuple = np.stack([np.squeeze(ntuple[feat]) for feat in branches], axis=1)
  n_entries = len(stack_ntuple)
  train_n_entries = int(train_fraction * n_entries)
  eval_n_entries = int(eval_fraction * n_entries)
  test_n_entries = n_entries - train_n_entries - eval_n_entries
  train_ntuples = stack_ntuple[:train_n_entries]
  eval_ntuples = stack_ntuple[train_n_entries:train_n_entries+eval_n_entries]
  test_ntuples = stack_ntuple[train_n_entries+eval_n_entries:]
  print(f'Total entries: {n_entries} train entries: {len(train_ntuples)}, test entries: {len(test_ntuples)}, eval entries: {len(eval_ntuples)}')

  # Scale up weights
  for weight_branch_name in weight_branch_names:
    weight_branch_index = branches.index(weight_branch_name)
    train_ntuples[:,weight_branch_index] = train_ntuples[:,weight_branch_index] / train_fraction
    test_ntuples[:,weight_branch_index] = test_ntuples[:,weight_branch_index] / test_fraction
    eval_ntuples[:,weight_branch_index] = eval_ntuples[:,weight_branch_index] / eval_fraction

  # Create the ntuples and tmp file
  out_train_ntuples = {}
  out_eval_ntuples = {}
  out_test_ntuples = {}
  for ibranch, branch in enumerate(branches):
    if branch == 'classID':
      out_train_ntuples[branch] = np.int32(train_ntuples[:,ibranch])
      out_eval_ntuples[branch] = np.int32(eval_ntuples[:,ibranch])
      out_test_ntuples[branch] = np.int32(test_ntuples[:,ibranch])
    else:
      out_train_ntuples[branch] = np.float32(train_ntuples[:,ibranch])
      out_eval_ntuples[branch] = np.float32(eval_ntuples[:,ibranch])
      out_test_ntuples[branch] = np.float32(test_ntuples[:,ibranch])
  tmp_filename = f'{output_filename}.tmp'
  with uproot.recreate(tmp_filename) as output_file:
    output_file["train_tree_nocut"] = out_train_ntuples
    output_file["test_tree_nocut"] = out_test_ntuples
    output_file["eval_tree_nocut"] = out_eval_ntuples
  print('Temporary saved trees to '+tmp_filename)

  # Apply cut using uproot
  in_file = uproot.open(tmp_filename)

  train_full_tree = in_file['train_tree_nocut']
  train_full_ntuples = train_full_tree.arrays(branches, library='np')
  if uproot_cut == '1': train_ntuples = train_full_tree.arrays(branches, library='np')
  else: train_ntuples = train_full_tree.arrays(branches, uproot_cut, library='np')

  test_full_tree = in_file['test_tree_nocut']
  test_full_ntuples = test_full_tree.arrays(branches, library='np')
  if uproot_cut == '1': test_ntuples = test_full_tree.arrays(branches, library='np')
  else: test_ntuples = test_full_tree.arrays(branches, uproot_cut, library='np')

  eval_full_tree = in_file['eval_tree_nocut']
  eval_full_ntuples = eval_full_tree.arrays(branches, library='np')
  if uproot_cut == '1': eval_ntuples = eval_full_tree.arrays(branches, library='np')
  else: eval_ntuples = eval_full_tree.arrays(branches, uproot_cut, library='np')

  #print(f'train_full: {train_full_ntuples}')
  #print(f'train: {train_ntuples}')
  with uproot.recreate(output_filename) as output_file:
    output_file["train_tree"] = train_ntuples
    output_file["train_tree_nocut"] = train_full_ntuples
    output_file["test_tree"] = test_ntuples
    output_file["test_tree_nocut"] = test_full_ntuples
    output_file["eval_tree"] = eval_ntuples
    output_file["eval_tree_nocut"] = eval_full_ntuples
  print('Tree saved to '+output_filename)
  in_file.close()

if __name__ == '__main__':
  start_time = time.time()

  np.random.seed(1)

  # Combine signal and background samples into one ntuple AND mix up signal and background event sequence.
  class_filenames = [
               [1,'ntuples/ggF_ntuples_GluGluHToZG_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8_2016APV201620172018.root'],
               [0,'ntuples/ggF_ntuples_ZGToLLG_01J_5f_lowMLL_lowGPt_TuneCP5_13TeV-amcatnloFXFX-pythia8_2016APV201620172018.root'],
               [0,'ntuples/ggF_ntuples_DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_2016APV201620172018.root']
               ]
  ntuple, branches = combine_ntuples(class_filenames)

  # Split sample into training, validation, and testing tree using event number. ([0,3,6,9]=training,[1,4,7]=validation,[2,5,8]=testing)
  split_ntuple(ntuple, branches, uproot_cut='(lly_m>120) & (lly_m<130)', weight_branch_names=['weightXyear','w_lumiXyear'], output_filename='ntuples/ggf_mva_ntuples.root')

  elapsed_time = time.time() - start_time
  print(f'Elapsed time: {elapsed_time}')
