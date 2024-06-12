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
  parser = argparse.ArgumentParser(description='Trains BDT usingTMVA')
  parser.add_argument('-i','--method_id', required=True)

  args = parser.parse_args()
  method_id = int(args.method_id)
  if method_id == 0: 
    mva_name = 'xgboost_decorr'
    train_tree_name = 'train_tree'
    weight_method = 0 # Weight signal and background equally
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_eta', 'l1_eta', 'l2_eta']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 1: 
    mva_name = 'xgboost_windowcut_hig19014'
    train_tree_name = 'train_tree_baseline'
    weight_method = 0 # Weight signal and background equally
    input_mva_ntuple = 'ntuples/ggf_mva_hig19014_decorr_ntuples.root'
    features = ['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'y_ptmass']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '(lly_m>120)&(lly_m<130)'
  if method_id == 2: 
    mva_name = 'xgboost_windowcut_decorr_hig19014'
    train_tree_name = 'train_tree_baseline'
    weight_method = 0 # Weight signal and background equally
    input_mva_ntuple = 'ntuples/ggf_mva_hig19014_decorr_ntuples.root'
    features = ['y_mva', 'yl_drmin', 'yl_drmax', 'cosThetamass3', 'costheta', 'phi', 'lly_ptmass', 'y_eta', 'l1_eta', 'l2_eta', 'l2_pt']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '(lly_m>120)&(lly_m<130)'
  if method_id == 3: 
    mva_name = 'xgboost_nowindowcut'
    train_tree_name = 'train_tree_baseline'
    weight_method = 0 # Weight signal and background equally
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 4: 
    mva_name = 'xgboost_windowcut'
    train_tree_name = 'train_tree'
    weight_method = 0 # Weight signal and background equally
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 5: 
    mva_name = 'xgboost_yptmass3'
    train_tree_name = 'train_tree'
    weight_method = 0 # Weight signal and background equally
    input_mva_ntuple = 'ntuples/ggf_mva_yptmass3_decorr_ntuples.root'
    features = ['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 6: 
    mva_name = 'xgboost_yptmass2'
    train_tree_name = 'train_tree'
    weight_method = 0 # Weight signal and background equally
    input_mva_ntuple = 'ntuples/ggf_mva_yptmass2_decorr_ntuples.root'
    features = ['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 7: 
    mva_name = 'xgboost_decorr_cosThetacut'
    train_tree_name = 'train_tree'
    weight_method = 0 # Weight signal and background equally
    input_mva_ntuple = 'ntuples/ggf_mva_cosTheta_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosTheta', 'costheta', 'phi', 'lly_ptmass', 'y_eta', 'l1_eta', 'l2_eta']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 8: 
    mva_name = 'xgboost_decorr19'
    train_tree_name = 'train_tree'
    weight_method = 0 # Weight signal and background equally
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'l1_ptmass', 'l2_pt', 'leplep_flavor', 'leplep_m', 'leplep_eta', 'lly_ptt', 'lly_eta', 'ht']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 9: 
    mva_name = 'xgboost_decorr_min19'
    train_tree_name = 'train_tree'
    weight_method = 0 # Weight signal and background equally
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'ht']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 10: 
    mva_name = 'xgboost_decorr_min19bad'
    train_tree_name = 'train_tree'
    weight_method = 0 # Weight signal and background equally
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'lly_ptt', 'ht']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 11: 
    mva_name = 'xgboost_min14_ypt'
    train_tree_name = 'train_tree'
    weight_method = 0 # Weight signal and background equally
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'ht', 'y_ptinfo2mass2']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 12: 
    mva_name = 'xgboost_min13'
    train_tree_name = 'train_tree'
    weight_method = 0 # Weight signal and background equally
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 13: 
    mva_name = 'xgboost_decorr18'
    train_tree_name = 'train_tree'
    weight_method = 0 # Weight signal and background equally
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'l1_ptmass', 'l2_pt', 'leplep_flavor', 'leplep_m', 'leplep_eta', 'lly_ptt', 'lly_eta']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 14: 
    mva_name = 'xgboost_min13_ypt'
    train_tree_name = 'train_tree'
    weight_method = 0 # Weight signal and background equally
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'y_ptinfo2mass2']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 15: 
    mva_name = 'xgboost19'
    train_tree_name = 'train_tree'
    weight_method = 0 # Weight signal and background equally
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'l1_ptmass', 'l2_pt', 'leplep_flavor', 'leplep_m', 'leplep_eta', 'lly_ptt', 'lly_eta','y_ptinfo2mass2']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 16: 
    mva_name = 'xgboost_tight_higwindow_var11'
    train_tree_name = 'train_tree'
    weight_method = 0 # Weight signal and background equally
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 17: 
    mva_name = 'xgboost_tight_higwindow_var12'
    train_tree_name = 'train_tree'
    weight_method = 0 # Weight signal and background equally
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'y_ptmass']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 18: 
    mva_name = 'xgboost_tight_higwindow_var12cosh'
    train_tree_name = 'train_tree'
    weight_method = 0 # Weight signal and background equally
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'y_ptinfo2mass2']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 19: 
    mva_name = 'xgboost_tight_higwindow_var13'
    train_tree_name = 'train_tree'
    weight_method = 0 # Weight signal and background equally
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 20: 
    mva_name = 'xgboost_tight_higwindow_var14'
    train_tree_name = 'train_tree'
    weight_method = 0 # Weight signal and background equally
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'npv']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 21: 
    mva_name = 'xgboost_tight0j_higwindow_var13'
    train_tree_name = 'train_tree'
    weight_method = 0 # Weight signal and background equally
    input_mva_ntuple = 'ntuples/ggf_tight0j_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 22: 
    mva_name = 'xgboost_tight1j_higwindow_var13'
    train_tree_name = 'train_tree'
    weight_method = 0 # Weight signal and background equally
    input_mva_ntuple = 'ntuples/ggf_tight1j_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 23: 
    mva_name = 'xgboost_tight1j_higwindow_var19'
    train_tree_name = 'train_tree'
    weight_method = 0 # Weight signal and background equally
    input_mva_ntuple = 'ntuples/ggf_tight1j_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'j1_pt', 'j1_eta', 'j1_phi', 'llyj_dphi', 'yj1_deta', 'yj1_dr', 'llyj1_ptbal']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 24: 
    mva_name = 'xgboost_ggfvbf_tight1j_higwindow_var19'
    train_tree_name = 'train_tree'
    weight_method = 0 # Weight signal and background equally
    input_mva_ntuple = 'ntuples/ggfvbf_tight1j_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'j1_pt', 'j1_eta', 'j1_phi', 'llyj_dphi', 'yj1_deta', 'yj1_dr', 'llyj1_ptbal']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 25: 
    mva_name = 'xgboost_ggfvbf_tight_higwindow_var13'
    train_tree_name = 'train_tree'
    weight_method = 0 # Weight signal and background equally
    input_mva_ntuple = 'ntuples/ggfvbf_tight_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 26: 
    mva_name = 'xgboost_ggfvbf_tight1j_higwindow_var13'
    train_tree_name = 'train_tree'
    weight_method = 0 # Weight signal and background equally
    input_mva_ntuple = 'ntuples/ggfvbf_tight1j_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 27: 
    mva_name = 'xgboost_ggfvbf_tight0j_higwindow_var13'
    train_tree_name = 'train_tree'
    weight_method = 0 # Weight signal and background equally
    input_mva_ntuple = 'ntuples/ggfvbf_tight0j_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 28: 
    mva_name = 'xgboost_ggfvbf_tight1j_higwindow_var15'
    train_tree_name = 'train_tree'
    weight_method = 0 # Weight signal and background equally
    input_mva_ntuple = 'ntuples/ggfvbf_tight1j_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 29: 
    mva_name = 'xgboost_ggfvbf_tight_higwindow_var15'
    train_tree_name = 'train_tree'
    weight_method = 0 # Weight signal and background equally
    input_mva_ntuple = 'ntuples/ggfvbf_tight_shape_decorr_weight_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 30: 
    mva_name = 'xgboost_wgt11_ggfvbf_tight_higwindow_var15'
    train_tree_name = 'train_tree'
    weight_method = 1 # Weight signal and background equally according to cross_section
    input_mva_ntuple = 'ntuples/ggfvbf_tight_shape_decorr_weight_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 31: 
    mva_name = 'xgboost_wgt11_ggfvbf_tight_higwindow_var13'
    train_tree_name = 'train_tree'
    weight_method = 1 # Weight signal and background equally according to cross_section
    input_mva_ntuple = 'ntuples/ggfvbf_tight_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 32: 
    mva_name = 'xgboost_wgt11_ggfvbf_tight0j_higwindow_var13'
    train_tree_name = 'train_tree'
    weight_method = 1 # Weight signal and background equally according to cross_section
    input_mva_ntuple = 'ntuples/ggfvbf_tight0j_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 33: 
    mva_name = 'xgboost_wgt11_ggfvbf_tight1j_higwindow_var15'
    train_tree_name = 'train_tree'
    weight_method = 1 # Weight signal and background equally according to cross_section
    input_mva_ntuple = 'ntuples/ggfvbf_tight1j_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 34: 
    mva_name = 'xgboost_wgt111_ggfvbf_tight_higwindow_var15'
    train_tree_name = 'train_tree'
    weight_method = 2 # Weight signal and background equally according to cross_section
    input_mva_ntuple = 'ntuples/ggfvbf_tight_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 35: 
    mva_name = 'xgboost_wgt111_ggfvbf_tight_higwindow_var13'
    train_tree_name = 'train_tree'
    weight_method = 2 # Weight ggf, vbf, and background equally according to cross_section
    input_mva_ntuple = 'ntuples/ggfvbf_tight_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 36: 
    mva_name = 'xgboost_wgt111_ggfvbf_tight0j_higwindow_var13'
    train_tree_name = 'train_tree'
    weight_method = 2 # Weight ggf, vbf, and background equally according to cross_section
    input_mva_ntuple = 'ntuples/ggfvbf_tight0j_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 37: 
    mva_name = 'xgboost_wgt111_ggfvbf_tight1j_higwindow_var15'
    train_tree_name = 'train_tree'
    weight_method = 2 # Weight ggf, vbf, and background equally according to cross_section
    input_mva_ntuple = 'ntuples/ggfvbf_tight1j_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 38: 
    mva_name = 'xgboost_wgt_ggfvbf_tight_higwindow_var15'
    train_tree_name = 'train_tree'
    weight_method = 3 # Weight signal and background according to cross_section
    input_mva_ntuple = 'ntuples/ggfvbf_tight_shape_decorr_weight_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 39: 
    mva_name = 'xgboost_wgt_ggfvbf_tight_higwindow_var13'
    train_tree_name = 'train_tree'
    weight_method = 3 # Weight ggf, vbf, and background according to cross_section
    input_mva_ntuple = 'ntuples/ggfvbf_tight_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 40: 
    mva_name = 'xgboost_wgt_ggfvbf_tight0j_higwindow_var13'
    train_tree_name = 'train_tree'
    weight_method = 3 # Weight ggf, vbf, and background according to cross_section
    input_mva_ntuple = 'ntuples/ggfvbf_tight0j_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 41: 
    mva_name = 'xgboost_wgt_ggfvbf_tight1j_higwindow_var15'
    train_tree_name = 'train_tree'
    weight_method = 3 # Weight ggf, vbf, and background according to cross_section
    input_mva_ntuple = 'ntuples/ggfvbf_tight1j_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 42: 
    mva_name = 'xgboost_wgtmass_ggfvbf_tight_higwindow_var15'
    train_tree_name = 'train_tree'
    weight_method = 4 # Weight signal and background according to signal mass
    input_mva_ntuple = 'ntuples/ggfvbf_tight_shape_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 43: 
    mva_name = 'xgboost_wgtmass11_ggfvbf_tight_higwindow_var15'
    train_tree_name = 'train_tree'
    weight_method = 5 # Weight signal and background according to signal mass. sum(w_sig) = sum(w_bkg)
    input_mva_ntuple = 'ntuples/ggfvbf_tight_shape_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 44: 
    mva_name = 'xgboost_wgtres11_ggfvbf_tight_higwindow_var15'
    train_tree_name = 'train_tree'
    weight_method = 6 # Weight signal and background according to 1/res. sum(w_sig) = sum(w_bkg)
    input_mva_ntuple = 'ntuples/ggfvbf_tight_shape_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 45: 
    mva_name = 'xgboost_wgtsigres11_ggfvbf_tight_higwindow_var15'
    train_tree_name = 'train_tree'
    weight_method = 7 # Weight signal and background according to 1/res. sum(w_sig) = sum(w_bkg)
    input_mva_ntuple = 'ntuples/ggfvbf_tight_shape_decorr_weight_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5}
    train_cut = '1'
  if method_id == 46: 
    mva_name = 'xgboost_wgtsigres211_ggfvbf_tight_higwindow_var15'
    train_tree_name = 'train_tree'
    weight_method = 8 # Weight signal and background according to 1/res^2. sum(w_sig) = sum(w_bkg)
    input_mva_ntuple = 'ntuples/ggfvbf_tight_shape_decorr_weight_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']
    bdt_settings = {'max_depth':4, 'learning_rate':0.1, 'n_estimators':500, 'min_child_weight':5, 'eval_metric':sklearn.metrics.log_loss}
    train_cut = '1'
  print(f'Training XGBoost with method {method_id}, {mva_name}')

  start_time = time.time()

  if weight_method == 0: spectators = ['lly_m', 'w_lumiXyear', 'weightXyear', 'classID', 'sampleID', 'wgteq1']
  elif weight_method == 1: spectators = ['lly_m', 'w_lumiXyear', 'weightXyear', 'classID', 'sampleID', 'wgt11']
  elif weight_method == 3: spectators = ['lly_m', 'w_lumiXyear', 'weightXyear', 'classID', 'sampleID', 'wgt']
  elif weight_method == 7: spectators = ['lly_m', 'w_lumiXyear', 'weightXyear', 'classID', 'sampleID', 'wgtsigres11']
  elif weight_method == 4: spectators = ['lly_m', 'w_lumiXyear', 'weightXyear', 'classID', 'sampleID', 'w_same_btos_shape']
  elif weight_method == 5: spectators = ['lly_m', 'w_lumiXyear', 'weightXyear', 'classID', 'sampleID', 'weightXyearXbtos_shape']
  elif weight_method == 6 or weight_method == 7 or weight_method == 8: spectators = ['lly_m', 'w_lumiXyear', 'weightXyear', 'classID', 'sampleID', 'lly_res']
  else: spectators = ['lly_m', 'w_lumiXyear', 'weightXyear', 'classID', 'sampleID']
  spectators.extend(['wgteq1', 'wgt', 'wgt11', 'wgtsigres11'])
  mass_index = 0
  w_lumiXyear_index = 1
  weightXyear_index = 2
  class_index = 3
  sample_index = 4
  train_weight_index = 5
  shape_index = 5

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

  if weight_method == 0: # Scales bkg to nsig_entries/nbkg_entries and sig to 1
    ## Weight signal and background equally
    #nbkg = np.count_nonzero(train_dataset.spec_array[:,class_index]==0)
    #nsig = np.count_nonzero(train_dataset.spec_array[:,class_index]==1)
    #nsig_nbkg = nsig * 1./ nbkg
    ## weight signal and bkg equally
    #train_weight_array = np.array(train_dataset.spec_array[:,class_index])
    #train_weight_array[train_weight_array == 0] = nsig_nbkg
    #print(train_weight_array,train_dataset.spec_array[:,len(spectators)-4])
    #print(f'nsig: {nsig} nbkg: {nbkg}. Reweight bkg by {nsig_nbkg} sum: {np.sum(train_weight_array)}')
    train_weight_array = train_dataset.spec_array[:,train_weight_index]
  elif weight_method == 1: # Weight signal and background equally according to cross_section
    ## Get sum of weights for background and signal
    #bkg_mask = train_dataset.spec_array[:,class_index]==0
    #bkg_sum = np.sum(train_dataset.spec_array[bkg_mask,weightXyear_index])
    #sig_mask = train_dataset.spec_array[:,class_index]==1
    #sig_sum = np.sum(train_dataset.spec_array[sig_mask,weightXyear_index])
    #bkg_over_sig = bkg_sum/sig_sum
    ## Scale signal to match background
    #train_weight_array = train_dataset.spec_array[:,weightXyear_index]
    #train_weight_array[sig_mask] = train_weight_array[sig_mask] * bkg_over_sig
    ## Set negative weights to 0
    #train_weight_array[train_weight_array<0] = 0
    #print(f'bkg events: {bkg_sum} signal events: {sig_sum} bkg/sig: {bkg_sum/sig_sum}')
    #print(train_weight_array, train_dataset.spec_array[:,len(spectators)-2])
    train_weight_array = train_dataset.spec_array[:,train_weight_index]
  elif weight_method == 2: # Weight ggf, vbf, and background equally according to cross_section
    # Get sum of weights for background and signal
    bkg_mask = train_dataset.spec_array[:,class_index]==0
    bkg_sum = np.sum(train_dataset.spec_array[bkg_mask,weightXyear_index])
    sig_ggf_mask = train_dataset.spec_array[:,sample_index]==4
    sig_ggf_sum = np.sum(train_dataset.spec_array[sig_ggf_mask,weightXyear_index])
    bkg_over_sig_ggf = bkg_sum/sig_ggf_sum
    sig_vbf_mask = train_dataset.spec_array[:,sample_index]==5
    sig_vbf_sum = np.sum(train_dataset.spec_array[sig_vbf_mask,weightXyear_index])
    bkg_over_sig_vbf = bkg_sum/sig_vbf_sum
    # Scale signal to match background
    train_weight_array = train_dataset.spec_array[:,weightXyear_index]
    train_weight_array[sig_ggf_mask] = train_weight_array[sig_ggf_mask] * bkg_over_sig_ggf
    train_weight_array[sig_vbf_mask] = train_weight_array[sig_vbf_mask] * bkg_over_sig_vbf
    # Set negative weights to 0
    train_weight_array[train_weight_array<0] = 0
    print(f'bkg events: {bkg_sum} signal ggf events: {sig_ggf_sum} bkg/sig(ggf): {bkg_over_sig_ggf} signal vbf events: {sig_vbf_sum} bkg/sig(vbf): {bkg_over_sig_vbf}')
  elif weight_method == 3: 
    #train_weight_array = train_dataset.spec_array[:,weightXyear_index]
    ## Set negative weights to 0
    #train_weight_array[train_weight_array<0] = 0
    train_weight_array = train_dataset.spec_array[:,train_weight_index]
  elif weight_method == 4: # Weight signal and background according to w_shape
    train_weight_array = train_dataset.spec_array[:,shape_index]
  elif weight_method == 5: # Weight signal and background according to w_shape. sum(w_sig) = sum(w_bkg)
    # Get sum of weights for background and signal
    bkg_mask = train_dataset.spec_array[:,class_index]==0
    bkg_sum = np.sum(train_dataset.spec_array[bkg_mask,shape_index])
    sig_mask = train_dataset.spec_array[:,class_index]==1
    sig_sum = np.sum(train_dataset.spec_array[sig_mask,shape_index])
    bkg_over_sig = bkg_sum/sig_sum
    # Scale signal to match background
    train_weight_array = train_dataset.spec_array[:,shape_index]
    train_weight_array[sig_mask] = train_weight_array[sig_mask] * bkg_over_sig
    # Set negative weights to 0
    train_weight_array[train_weight_array<0] = 0
  elif weight_method == 6: # Weight signal and background according to w_shape. sum(w_sig) = sum(w_bkg)
    #print(train_dataset.spec_array[:,shape_index], 1./train_dataset.spec_array[:,shape_index], train_dataset.spec_array[:,weightXyear_index], 1./train_dataset.spec_array[:,shape_index]*train_dataset.spec_array[:,weightXyear_index])
    # Get sum of weights for background and signal
    bkg_mask = train_dataset.spec_array[:,class_index]==0
    bkg_sum = np.sum(train_dataset.spec_array[bkg_mask,weightXyear_index]/train_dataset.spec_array[bkg_mask,shape_index])
    sig_mask = train_dataset.spec_array[:,class_index]==1
    sig_sum = np.sum(train_dataset.spec_array[sig_mask,weightXyear_index]/train_dataset.spec_array[sig_mask,shape_index])
    bkg_over_sig = bkg_sum/sig_sum
    # Scale signal to match background
    train_weight_array = train_dataset.spec_array[:,weightXyear_index]/train_dataset.spec_array[:,shape_index]
    train_weight_array[sig_mask] = train_weight_array[sig_mask] * bkg_over_sig
    # Set negative weights to 0
    train_weight_array[train_weight_array<0] = 0
  elif weight_method == 7: # Weight signal and background according to w_shape. sum(w_sig) = sum(w_bkg)
    ## Get sum of weights for background and signal
    #bkg_mask = train_dataset.spec_array[:,class_index]==0
    #bkg_sum = np.sum(train_dataset.spec_array[bkg_mask,weightXyear_index])
    #sig_mask = train_dataset.spec_array[:,class_index]==1
    #sig_sum = np.sum(train_dataset.spec_array[sig_mask,weightXyear_index]/train_dataset.spec_array[sig_mask,shape_index])
    #bkg_over_sig = bkg_sum/sig_sum
    ## Scale signal to match background
    #train_weight_array = train_dataset.spec_array[:,weightXyear_index]
    #train_weight_array[sig_mask] = train_dataset.spec_array[sig_mask,weightXyear_index]/train_dataset.spec_array[sig_mask,shape_index] * bkg_over_sig
    ## Set negative weights to 0
    #train_weight_array[train_weight_array<0] = 0
    #print(train_weight_array, train_dataset.spec_array[:,len(spectators)-1])
    train_weight_array = train_dataset.spec_array[:,train_weight_index]
  elif weight_method == 8: # Weight signal and background according to w_shape. sum(w_sig) = sum(w_bkg)
    # Get sum of weights for background and signal
    bkg_mask = train_dataset.spec_array[:,class_index]==0
    bkg_sum = np.sum(train_dataset.spec_array[bkg_mask,weightXyear_index])
    sig_mask = train_dataset.spec_array[:,class_index]==1
    sig_sum = np.sum(train_dataset.spec_array[sig_mask,weightXyear_index]/np.power(train_dataset.spec_array[sig_mask,shape_index],2))
    bkg_over_sig = bkg_sum/sig_sum
    # Scale signal to match background
    train_weight_array = train_dataset.spec_array[:,weightXyear_index]
    train_weight_array[sig_mask] = train_dataset.spec_array[sig_mask,weightXyear_index]/np.power(train_dataset.spec_array[sig_mask,shape_index],2) * bkg_over_sig
    ## Set negative weights to 0
    #train_weight_array[train_weight_array<0] = 0

  nlabels = train_hot_label_array.shape[1]

  # Train xgboost
  print("Training xgboost")
  xgbdt_classifier = xgboost.XGBClassifier(**bdt_settings)
  xgbdt_classifier.fit(train_feature_array, train_hot_label_array[:,nlabels-1], sample_weight=train_weight_array)

  # Save xgboost model
  xgbdt_classifier.save_model(f'mva_output/{mva_name}_xgboost_model.json')
  print(f'Saved xgboost model to mva_output/{mva_name}_xgboost_model.json')
  # Load xgboost model
  xgbdt_classifier.load_model(f'mva_output/{mva_name}_xgboost_model.json')

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
