#!/usr/bin/env python3
import torch
import uproot
import numpy as np
import copy
class RootDataset(torch.utils.data.Dataset):
  def __init__(self, root_filename, tree_name, features, cut, spectators, class_branch, entry_stop=None, entry_start=None, normalize=None, transform=None):
    self.root_file = uproot.open(root_filename)
    self.tree = self.root_file[tree_name]
    self.transform = transform
    # feature_array = {feat1_name : [[feat1_event0, feat1_event2, ...]], } # for cut='1'
    # feature_array = {feat1_name : [feat1_event0, feat1_event2, ...], } # for cut!='1'
    feature_array = self.tree.arrays(features,
                                cut,
                                library='np')
    # self.feature_array = [ (feat1_event0, feat2_event0, ...),  ]
    if cut == '1': self.feature_array = np.stack([feature_array[feat][0] for feat in features], axis=1)
    else: self.feature_array = np.stack([feature_array[feat] for feat in features], axis=1)
    spec_array = self.tree.arrays(spectators,
                             cut,
                             library='np')
    # self.spec_array = [ (spec1_event0, spec2_event0, ...),  ]
    if cut == '1': self.spec_array = np.stack([spec_array[spec][0] for spec in spectators], axis=1)
    else: self.spec_array = np.stack([spec_array[spec] for spec in spectators], axis=1)
    # label_array = {classId: [1, 1, 0, ... ]}
    label_array = self.tree.arrays(class_branch,
                              cut,
                              library='np')
    # label_hotencoding = [ (0, 1), (0, 1), ... ]
    if cut == '1': label_array = label_array['classID'][0]
    else: label_array = label_array['classID']
    label_hotencoding = np.zeros((label_array.size, label_array.max()+1))
    label_hotencoding[np.arange(label_array.size), label_array] = 1
    self.label_array = np.array(label_hotencoding, dtype=int)

    # remove unlabeled data
    self.feature_array = self.feature_array[np.sum(self.label_array, axis=1) == 1]
    self.spec_array = self.spec_array[np.sum(self.label_array, axis=1) == 1]
    self.label_array = self.label_array[np.sum(self.label_array, axis=1) == 1]

    # normalize
    # TODO Clip min max
    if normalize:
      feat_min = np.amin(self.feature_array,0)
      feat_max = np.amax(self.feature_array,0)
      for ifeat, [min_x, max_x] in enumerate(normalize):
        #print(f'[Info] ifeat[{ifeat}] data min: {feat_min[ifeat]} max: {feat_max[ifeat]} norm min: {min_x} max: {max_x}')
        self.feature_array[:,ifeat] = 2.*(self.feature_array[:,ifeat]-min_x)/(max_x-min_x) - 1.

    # Split data
    if entry_stop and entry_stop:
      self.feature_array = self.feature_array[entry_start:entry_stop]
      self.spec_array = self.spec_array[entry_start:entry_stop]
      self.label_array = self.label_array[entry_start:entry_stop]
    elif entry_stop:
      self.feature_array = self.feature_array[:entry_stop]
      self.spec_array = self.spec_array[:entry_stop]
      self.label_array = self.label_array[:entry_stop]
    elif entry_start:
      self.feature_array = self.feature_array[entry_start:]
      self.spec_array = self.spec_array[entry_start:]
      self.label_array = self.label_array[entry_start:]

  def __len__(self):
    return len(self.label_array)

  def __getitem__(self, idx):
    #sample = {'feature_array': self.feature_array[idx], 'label_array': self.label_array[idx], 'spec_array': self.spec_array[idx]}
    sample = [self.feature_array[idx], self.label_array[idx], self.spec_array[idx]]
    if self.transform:
      sample = self.transform(sample)
    return sample

def unnormalize(values, norm_weights):
  feature_array = copy.deepcopy(values)
  for ifeat, [min_x, max_x] in enumerate(norm_weights):
    feature_array[:,ifeat] = (values[:,ifeat]+1)*(max_x-min_x)*1./2 + min_x
  return feature_array

if __name__ == '__main__':
  print("Example of using RootDataset")

  example_filename = 'data/example_data.root'
  features = ['min_dR', 'max_dR', 'pt_mass', 'cosTheta', 'costheta', 
              'llg_mass_err', 'phi', 
              'photon_rapidity', 'l1_rapidity', 'l2_rapidity',
              'llg_flavor', 'llg_ptt', 
              'photon_pt_mass', 'lead_lep_pt', 'sublead_lep_pt']
  normalize_max_min = [[0.4,3.7], [0.4,5.0], [0.0,11.0], [-1.0, 1.0], [-1.0, 1.0],
                      [1.1, 3.4], [0.0, 6.3],
                      [-2.5, 2.5], [-2.5, 2.5], [-2.5, 2.5],
                      [0., 1.], [0.0, 400.], 
                      [0.13, 2.0], [25.0, 1000.], [15.0, 500.]]
  example_dataset = RootDataset(root_filename= example_filename,
                            tree_name = "eval_full_tree",
                            features = features,
                            normalize = normalize_max_min,
                            cut = '1',
                            spectators = ['llg_mass'],
                            class_branch = ['classID'])

  print(f'Entries: {len(example_dataset)}')
  print(f'Features: {features}')
  print(f'Feature values: {unnormalize(example_dataset.feature_array, normalize_max_min)}')
  print(f'Normalized feature values: {example_dataset.feature_array}')
  print(f'Labels: {example_dataset.label_array[:,1]}')
  print(f'Spectators: {example_dataset.spec_array}')
