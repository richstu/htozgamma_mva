#!/usr/bin/env python3
import ROOT
import sklearn.metrics
import sklearn.ensemble
import xgboost
import uproot
import numpy as np
import argparse
import time
import shap
import matplotlib.pyplot as plt

import sys
sys.path.append('../python_scripts')
from RootDataset import RootDataset

def xgboost_shap_importance(root_filename, tree_name, features, xgboost_predict, output_tag, nevents=100):
  print("Using SHAP to evaluate feature importance")
  mva_file = uproot.open(root_filename)
  feature_array = mva_file[tree_name].arrays(features,'1',library='np')
  feature_array = np.stack([feature_array[feat][0] for feat in features], axis=1) # (nentries, nfeatures)
  #feature_array = mva_file[tree_name].arrays('x',library='np')['x']
  def infer_mva(feature_array):
    xgboost_raw_result = xgboost_predict(feature_array)
    xgboost_result = xgboost_raw_result[:,1]
    return xgboost_result
  explainer = shap.Explainer(infer_mva, feature_array[:nevents])
  shap_values = explainer(feature_array[:nevents])
  # Make shap bar plot
  plt.figure(1)
  plt_ax = shap.plots.bar(shap_values, show=False, max_display=20)
  # Change y axis label names
  y_labels = plt_ax.get_yticklabels()
  for y_label in y_labels:
    label_index = int(y_label.get_text().replace('Feature ',''))
    y_label.set_text(features[label_index])
  plt_ax.set_yticklabels(y_labels)
  # Save plot
  figure_name = f'plots/{output_tag}_shap.pdf'
  plt.savefig(figure_name, bbox_inches='tight')
  print(f'Plot saved to {figure_name}')
  # Make shap bee plot
  plt.figure(2)
  plt_ax = shap.plots.beeswarm(shap_values, show=False, max_display=20)
  # Change y axis label names
  y_labels = plt_ax.get_yticklabels()
  for y_label in y_labels:
    label_index = int(y_label.get_text().replace('Feature ',''))
    y_label.set_text(features[label_index])
  plt_ax.set_yticklabels(y_labels)
  # Save plot
  figure_name = f'plots/{output_tag}_shap_bee.pdf'
  plt.savefig(figure_name, bbox_inches='tight')
  print(f'Plot saved to {figure_name}')

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Trains BDT using TMVA')
  parser.add_argument('-i','--method_id', help='Train method: 0:standard_xgboost, 1:standard_xgboost_hig19014', default=0)

  args = parser.parse_args()
  method_id = int(args.method_id)
  if method_id == 0: 
    mva_name = 'standard_xgboost'
    train_tree_name = 'train_tree'
    weight_method = 0 # Weight signal and background equally
    input_mva_ntuple = 'ntuples/vbf_mva_decorr_ntuples.root'
    features = ['y_mva','yl_drmin','yl_drmax','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'lly_ptt', 'jj_deta', 'jj_dphi', 'yj1_dr', 'yj2_dr', 'llyjj_dphi', 'j1_pt', 'j2_pt', 'llyjj_ptbal', 'yjj_zep']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  elif method_id == 1: 
    mva_name = 'standard_xgboost_hig19014'
    weight_method = 0 # Weight signal and background equally
    train_tree_name = 'train_tree_baseline'
    input_mva_ntuple = 'ntuples/vbf_mva_hig19014_decorr_ntuples.root'
    features = ['y_mva','yl_drmin','yl_drmax','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'lly_ptt', 'jj_deta', 'jj_dphi', 'yj1_dr', 'yj2_dr', 'llyjj_dphi', 'j1_pt', 'j2_pt', 'llyjj_ptbal', 'yjj_zep']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  print(f'Training XGBoost with method {method_id}, {mva_name}')

  start_time = time.time()

  spectators = ['lly_m', 'w_lumiXyear', 'weightXyear', 'classID', 'sampleID']
  mass_index = 0
  w_lumiXyear_index = 1
  weightXyear_index = 2
  class_index = 3
  sample_index = 4

  train_dataset = RootDataset(root_filename= input_mva_ntuple,
                            tree_name = train_tree_name,
                            features = features,
                            cut = train_cut,
                            spectators = spectators,
                            class_branch = ['classID'])
  print(f'train entries: {len(train_dataset)}')

  train_feature_array = train_dataset.feature_array
  train_hot_label_array = train_dataset.label_array
  train_label_array = train_dataset.label_array[:,1]
  train_spec_array = train_dataset.spec_array
  train_mass_array = train_dataset.spec_array[:,mass_index]

  if weight_method == 0:
    # Weight signal and background equally
    nbkg = np.count_nonzero(train_dataset.spec_array[:,class_index]==0)
    nsig = np.count_nonzero(train_dataset.spec_array[:,class_index]==1)
    nsig_nbkg = nsig * 1./ nbkg
    # weight signal and bkg equally
    train_weight_array = np.array(train_dataset.spec_array[:,class_index])
    train_weight_array[train_weight_array == 0] = nsig_nbkg
    print(f'nsig: {nsig} nbkg: {nbkg}. Reweight bkg by {nsig_nbkg} sum: {np.sum(train_weight_array)}')

  nlabels = train_hot_label_array.shape[1]

  # Train xgboost
  print("Training xgboost")
  xgbdt_classifier = xgboost.XGBClassifier(**bdt_settings)
  xgbdt_classifier.fit(train_feature_array, train_hot_label_array[:,nlabels-1], sample_weight=train_weight_array)

  # Save xgboost model
  xgbdt_classifier.save_model('mva_output/{mva_name}_xgboost_model.json')
  # Load xgboost model
  xgbdt_classifier.load_model('mva_output/{mva_name}_xgboost_model.json')

  ## Run trained bdt over full set
  print("Evaluating xgboost")
  tree_names = [str(key.split(';')[0]) for key in uproot.open(input_mva_ntuple).keys()]
  infer_output_filename = f'mva_output/{mva_name}_results.root'
  xgbt_root_file = uproot.recreate(infer_output_filename)
  for tree_name in tree_names:
    eval_dataset = RootDataset(root_filename=input_mva_ntuple,
                              tree_name = tree_name,
                              features = features,
                              cut = '1',
                              spectators = spectators,
                              class_branch = ['classID'])
    print(f'{tree_name} entries: {len(eval_dataset)}')
    eval_feature_array = eval_dataset.feature_array # (entries, nfeature)
    eval_hot_label_array = eval_dataset.label_array
    eval_label_array = eval_dataset.label_array[:,1]
    eval_spec_array = eval_dataset.spec_array
    eval_mass_array = eval_dataset.spec_array[:,mass_index]
    eval_weight_array = eval_dataset.spec_array[:,weightXyear_index]
    eval_w_lumi_array = eval_dataset.spec_array[:,w_lumiXyear_index]
    eval_sampleID_array = eval_dataset.spec_array[:,sample_index]

    eval_predict_array_xgbdt_raw = xgbdt_classifier.predict_proba(eval_feature_array)
    eval_predict_array_xgbdt = eval_predict_array_xgbdt_raw[:,1]

    # Make dict for saving
    output_dict = {}
    # Make feature dict
    for ifeature, feature_name in enumerate(features):
      output_dict[feature_name] = eval_feature_array[:,ifeature]
    # Add other branches
    output_dict['y'] = eval_label_array
    output_dict['yhat'] = eval_predict_array_xgbdt
    output_dict['mass'] = eval_mass_array
    output_dict['weight'] = eval_weight_array
    output_dict['w_lumi'] = eval_weight_array
    output_dict['sampleID'] = eval_sampleID_array

    # Save to root file
    xgbt_root_file[tree_name] = output_dict

  print('Wrote xgbdt results to '+infer_output_filename)

  # Use SHAP to evaluate feature importance
  xgboost_shap_importance(infer_output_filename, 'train_tree', features, xgbdt_classifier.predict_proba, mva_name)

  elapsed_time = time.time() - start_time
  print(f'Training time: {elapsed_time}')
