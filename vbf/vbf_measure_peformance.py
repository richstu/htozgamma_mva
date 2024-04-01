#!/usr/bin/env python3
import ROOT
ROOT.gROOT.SetBatch(ROOT.kTRUE)
import math
import numpy as np
import sklearn.metrics
import uproot
import matplotlib.pyplot as plt
import array
import ctypes
import re
import os
import shap
from ctypes import c_double

import sys
sys.path.append('../python_scripts')
import measure_tools


if __name__ == '__main__':

  signi_detail_dict = {}
  train_signi_detail_dict = {}
  roc_detail_dict = {}
  train_roc_detail_dict = {}
  bkg_mass_difference_detail_dict = {}
  sig_mass_difference_detail_dict = {}
  bkg_feature_difference_detail_dict = {}
  sig_feature_difference_detail_dict = {}
  bkg_feature_difference_in_bin_detail_dict = {}
  sig_feature_difference_in_bin_detail_dict = {}
  overtrain_detail_dict = {}
  binned_signi_detail_dict = {}
  train_binned_signi_detail_dict = {}
  auc_sci_detail_dict = {}
  train_auc_sci_detail_dict = {}

  # Match mva_name to name used in training to make summary output pdfs
  # mva_info[mva_name] = [mva_result root filename, dict of information]
  mva_info = {\
    #'standard_tmva_bdt':['mva_output/standard_tmva_bdt_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva','yl_drmin','yl_drmax','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'lly_ptt', 'jj_deta', 'jj_dphi', 'yj1_dr', 'yj2_dr', 'llyjj_dphi', 'j1_pt', 'j2_pt', 'llyjj_ptbal', 'yjj_zep']}],
  }

  # Measure performance
  for mva_name in mva_info:
    mva_filename = mva_info[mva_name][0]
    branches = mva_info[mva_name][1]

    integrated_significance, integrated_significance_err, signi_detail = measure_tools.calculate_significance(mva_filename, 'eval_tree_baseline', branches, fixed_width=False, detail_output=True)
    print(f'[{mva_name}] Total significance: {integrated_significance:.3f}+-{integrated_significance_err:.3f}')
    train_integrated_significance, train_integrated_significance_err, train_signi_detail = measure_tools.calculate_significance(mva_filename, 'train_tree_baseline', branches, fixed_width=False, detail_output=True)
    print(f'[{mva_name}] Total significance (train): {train_integrated_significance:.3f}+-{train_integrated_significance_err:.3f}')
    auc, roc_detail = measure_tools.calculate_roc(mva_filename, 'eval_tree', branches, narrow_mass_window=True, detail_output=True)
    print(f'[{mva_name}] AUC: {auc:.1f}%')
    train_auc, train_roc_detail = measure_tools.calculate_roc(mva_filename, 'train_tree', branches, narrow_mass_window=True, detail_output=True)
    print(f'[{mva_name}] AUC(train): {train_auc:.1f}%')
    bkg_mass_shape_difference, bkg_mass_difference_detail = measure_tools.calculate_mass_shape_difference(mva_filename, 'eval_tree_baseline', branches, detail_output=True)
    print(f'[{mva_name}] Mass shape difference: {bkg_mass_shape_difference:.4f}')
    obs_mva_correlation = measure_tools.calculate_obs_mva_correlation(mva_filename, 'eval_tree_baseline', branches)
    print(f'[{mva_name}] Observable-MVA correlation: {obs_mva_correlation:.3f}')
    pvalue_signal, pvalue_bkg, overtrain_detail = measure_tools.calculate_overtraining(mva_filename, 'train_tree', 'eval_tree', branches, narrow_mass_window=True, detail_output=True)
    print(f'[{mva_name}] chi-2 pvalue (test,eval): signal: {pvalue_signal:.4f} background: {pvalue_bkg:.4f}')
    binned_significance, binned_significance_err, binned_signi_detail = measure_tools.calculate_binned_significance(mva_filename, 'eval_tree_baseline', branches, detail_output=True, mva_nbins=4, throw_away_bin=0)
    print(f'[{mva_name}] Significance 4 bins: {binned_significance:.3f} +- {binned_significance_err:.3f}')
    train_binned_significance, train_binned_significance_err, train_binned_signi_detail = measure_tools.calculate_binned_significance(mva_filename, 'train_tree_baseline', branches, detail_output=True, mva_nbins=4, throw_away_bin=0)
    print(f'[{mva_name}] Significance 4 bins (train): {train_binned_significance:.3f} +- {train_binned_significance_err:.3f}')
    auc_sci, auc_sci_detail = measure_tools.calculate_auc_sci(mva_filename, 'eval_tree', branches, detail_output=True)
    print(f'[{mva_name}] AUC: {auc_sci[0]*100:.1f}% CSI: {auc_sci[1]:.2f}')
    train_auc_sci, train_auc_sci_detail = measure_tools.calculate_auc_sci(mva_filename, 'train_tree', branches, detail_output=True)
    print(f'[{mva_name}] AUC (train): {auc_sci[0]*100:.1f}% CSI (train): {auc_sci[1]:.2f}')
    # Calculation for plotting
    sig_mass_shape_difference, sig_mass_difference_detail = measure_tools.calculate_mass_shape_difference(mva_filename, 'eval_tree_baseline', branches, y_value=1,detail_output=True)
    bkg_feature_shape_difference = measure_tools.find_feature_shape_difference(mva_filename, 'eval_tree_baseline', branches, y_value=0)
    sig_feature_shape_difference = measure_tools.find_feature_shape_difference(mva_filename, 'eval_tree_baseline', branches, y_value=1)
    bkg_feature_shape_difference_in_bin = measure_tools.find_feature_shape_difference_in_bins(mva_filename, 'eval_tree_baseline', branches, mva_bins=binned_signi_detail[1][3][1], y_value=0)
    sig_feature_shape_difference_in_bin = measure_tools.find_feature_shape_difference_in_bins(mva_filename, 'eval_tree_baseline', branches, mva_bins=train_binned_signi_detail[1][3][1], y_value=1)

    # Direct plotting
    measure_tools.plot_feature_shapes(mva_filename, 'train_tree_baseline', branches['observable'], branches["x"], branches['weight'], f'{branches["y"]}==1', f'{branches["y"]}==0', f'{mva_name}_bkg')
    #measure_tools.plot_feature_shapes(mva_filename, 'train_tree_baseline', branches['observable'], branches["x"], branches['weight'], f'{branches["y"]}==1', f'{branches["y"]}==1', f'{mva_name}_sig')

    # Collect detail for plots
    signi_detail_dict[mva_name] = signi_detail
    train_signi_detail_dict[mva_name] = train_signi_detail
    roc_detail_dict[mva_name] = roc_detail
    train_roc_detail_dict[mva_name] = train_roc_detail
    bkg_mass_difference_detail_dict[mva_name] = bkg_mass_difference_detail
    sig_mass_difference_detail_dict[mva_name] = sig_mass_difference_detail
    bkg_feature_difference_detail_dict[mva_name] = bkg_feature_shape_difference
    sig_feature_difference_detail_dict[mva_name] = sig_feature_shape_difference
    bkg_feature_difference_in_bin_detail_dict[mva_name] = bkg_feature_shape_difference_in_bin
    sig_feature_difference_in_bin_detail_dict[mva_name] = sig_feature_shape_difference_in_bin
    overtrain_detail_dict[mva_name] = overtrain_detail
    binned_signi_detail_dict[mva_name] = binned_signi_detail
    train_binned_signi_detail_dict[mva_name] = train_binned_signi_detail
    auc_sci_detail_dict[mva_name] = auc_sci_detail
    train_auc_sci_detail_dict[mva_name] = train_auc_sci_detail

  # Make plots using detail
  measure_tools.draw_signi_detail(signi_detail_dict)
  measure_tools.draw_signi_detail_train_eval(signi_detail_dict, train_signi_detail_dict)
  measure_tools.draw_roc_detail(roc_detail_dict)
  measure_tools.draw_roc_detail_train_eval(roc_detail_dict, train_roc_detail_dict)
  measure_tools.draw_mass_shape_difference(bkg_mass_difference_detail_dict,name_tag='bkg')
  measure_tools.draw_mass_shape_difference(sig_mass_difference_detail_dict,name_tag='sig')
  measure_tools.draw_overtraining(overtrain_detail_dict)
  measure_tools.draw_binned_signi_detail(binned_signi_detail_dict)
  measure_tools.draw_binned_signi_detail(train_binned_signi_detail_dict,name_tag='train')
  measure_tools.draw_binned_signi_detail_train_eval(binned_signi_detail_dict, train_binned_signi_detail_dict)
  measure_tools.draw_auc_sci_detail(auc_sci_detail_dict)
  measure_tools.draw_sci_detail_train_eval(auc_sci_detail_dict, train_auc_sci_detail)
  measure_tools.draw_feature_shape_difference(bkg_feature_difference_detail_dict,name_tag='bkg_8bin')
  measure_tools.draw_feature_shape_difference(sig_feature_difference_detail_dict,name_tag='sig_8bin')
  measure_tools.draw_feature_shape_difference(bkg_feature_difference_in_bin_detail_dict,name_tag='bkg_4bin')
  measure_tools.draw_feature_shape_difference(sig_feature_difference_in_bin_detail_dict,name_tag='sig_4bin')

  # Collect results into one pdf
  for mva_name in mva_info:

    # Input distributions
    for ipage in range(int(math.ceil(len(mva_info[mva_name][1]['x'])/6))):
      input_branches = ''
      for ibranch, input_branch in enumerate(mva_info[mva_name][1]['x']): 
        if ibranch >= ipage*6 and ibranch < (ipage+1)*6: input_branches += f'plots/{mva_name}_bkg_{measure_tools.slugify(input_branch)}_bins.pdf plots/{mva_name}_bkg_{mva_info[mva_name][1]["observable"]}_{measure_tools.slugify(input_branch)}.pdf '
      command = f'pdf_combine.py -f -i {input_branches} -x 4 -y 3 -o plots/summary_{mva_name}_inputs_{ipage}.pdf'
      os.system(command)

    # Validation metrics
    command = f'pdf_combine.py -f -i plots/{mva_name}_shap.pdf plots/{mva_name}_shap_bee.pdf -x 2 -y 1 -o plots/summary_{mva_name}_validation_1.pdf'
    os.system(command)
    input_branches = ''
    for input_branch in mva_info[mva_name][1]['x']: input_branches += f'plots/{mva_name}_bkg_8bin_hist_mva_{measure_tools.slugify(input_branch)}.pdf '
    command = f'pdf_combine.py -f -i {input_branches} -x 5 -y 3 -o plots/summary_{mva_name}_validation_2.pdf'
    os.system(command)
    input_branches = ''
    for input_branch in mva_info[mva_name][1]['x']: input_branches += f'plots/{mva_name}_bkg_4bin_hist_mva_{measure_tools.slugify(input_branch)}.pdf '
    command = f'pdf_combine.py -f -i {input_branches} -x 5 -y 3 -o plots/summary_{mva_name}_validation_3.pdf'
    os.system(command)

    # Significance/ROC and overtraining metrics
    command = f'pdf_combine.py -f -i plots/{mva_name}_roc.pdf plots/{mva_name}_significances_with_resolution.pdf plots/{mva_name}_binned_significances.pdf plots/{mva_name}_sig_bkg_overtrain.pdf plots/{mva_name}_binned_mva_per_sample.pdf plots/{mva_name}_binned_mva_per_sample_train.pdf'
    command += f' -x 3 -y 2 -o plots/summary_{mva_name}_significance.pdf'
    os.system(command)

    # Final mass plots
    command = f'pdf_combine.py -f -i plots/{mva_name}_binned_mva_mlly_bin0_per_sample.pdf plots/{mva_name}_binned_mva_mlly_bin1_per_sample.pdf plots/{mva_name}_binned_mva_mlly_bin2_per_sample.pdf plots/{mva_name}_binned_mva_mlly_bin3_per_sample.pdf plots/{mva_name}_binned_mva_per_sample.pdf plots/{mva_name}_bkg_hist_mva.pdf plots/{mva_name}_sig_hist_mva.pdf plots/{mva_name}_signal_resolution.pdf'
    command += f' -x 4 -y 2 -o plots/summary_{mva_name}_mass_shape.pdf'
    os.system(command)
