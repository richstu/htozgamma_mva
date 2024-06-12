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
import argparse

import sys
sys.path.append('../python_scripts')
import measure_tools

# rows = [[value, value, ...], next row]
# row_align = 'cccc'
def make_tex_table(rows, row_align, output_filename):
  tex_string = r'''\documentclass[10pt,oneside]{report}
\usepackage{graphicx,xspace,amssymb,amsmath,colordvi,colortbl,verbatim,multicol}
\usepackage{multirow, rotating}

\usepackage[active,tightpage]{preview}

\usepackage{siunitx}
\sisetup{round-mode = figures, round-precision=2}

\renewcommand{\arraystretch}{1.1}

\begin{document}
\begin{preview}
'''
  tex_string += f'\\begin{{tabular}}{{ {row_align} }}\n'
  for row in rows:
    for item in row:
      tex_string += f'{item} &'
    tex_string = tex_string[:-1]+'\\\\\n'
  tex_string += r'\end{tabular}'
  tex_string += r'''\end{preview}
\end{document}
'''
  with open(output_filename,'w') as output_file:
    output_file.write(tex_string)
  print(f'Wrote to {output_filename}')

def make_mva_table(binned_signi_detail):
  mva_nbins, binned_signi_detail_list, sig_hist, bkg_hist, sig_obs_hists, bkg_obs_hists, mva_hists, obs_hists, sig_ids, bkg_ids = binned_signi_detail
  nbins, bin_edges, significance, significance_err, significances, significance_errs, signal_yield, background_yield = binned_signi_detail_list[3]
  #print(bin_edges,  significance, significance_err, significances, significance_errs, signal_yield, background_yield)
  # Get sample yields based on bin_edges
  #bin_yields[sample_id][ibin] = [unweighted, weighted, bin_cut]
  bin_yields = {}
  chain = ROOT.TChain('eval_tree_baseline')
  chain.Add(mva_filename)
  significance_window_cut = f'{branches["observable"]}>120&&{branches["observable"]}<130'
  hist_name = f'hist_{branches["observable"]}'
  hist = ROOT.TH1D(hist_name, hist_name, 40, 120, 130)
  # Loop over each sample
  for sample_id in branches['bkg_ids']+branches['sig_ids']:
    sample_cut = f'{branches["sample_id"]}=={sample_id}'
    if sample_id not in bin_yields: bin_yields[sample_id] = {}
    # Loop over each bin
    for ibin in range(nbins):
      if ibin==0: bin_cut = f'{branches["yhat"]}<{bin_edges[0]}'
      elif ibin==nbins-1: bin_cut = f'{branches["yhat"]}>={bin_edges[-1]}'
      else: bin_cut = f'{branches["yhat"]}>={bin_edges[ibin-1]}&&{branches["yhat"]}<{bin_edges[ibin]}'
      # Get yield
      chain.Draw(f'{branches["observable"]}>>{hist_name}', f'({significance_window_cut}&&{sample_cut}&&{bin_cut})*{branches["weight"]}', 'goff')
      bin_yields[sample_id][ibin] = [hist.GetEntries(), hist.GetSum(), bin_cut.replace('&&',r'\&')]
  # Make tex table using yields
  bin_table = []
  bin_table.append(['MVA bin', 'MVA cut', 'Significance', 'SMZG', 'DYJets', 'GGF'])
  bin_table.append(['1', f'${bin_yields[1][3][2]}$', f'{significances[3]:.3f}+-{significance_errs[3]:.3f}', f'{bin_yields[1][3][1]:.3f}', f'{bin_yields[2][3][1]:.3f}', f'{bin_yields[3][3][1]:.3f}'])
  bin_table.append(['', '', 'no weights', bin_yields[1][3][0], bin_yields[2][3][0], bin_yields[3][3][0]])
  bin_table.append(['2', f'${bin_yields[1][2][2]}$', f'{significances[2]:.3f}+-{significance_errs[2]:.3f}', f'{bin_yields[1][2][1]:.3f}', f'{bin_yields[2][2][1]:.3f}', f'{bin_yields[3][2][1]:.3f}'])
  bin_table.append(['', '', 'no weights', bin_yields[1][2][0], bin_yields[2][2][0], bin_yields[3][2][0]])
  bin_table.append(['3', f'${bin_yields[1][1][2]}$', f'{significances[1]:.3f}+-{significance_errs[1]:.3f}', f'{bin_yields[1][1][1]:.3f}', f'{bin_yields[2][1][1]:.3f}', f'{bin_yields[3][1][1]:.3f}'])
  bin_table.append(['', '', 'no weights', bin_yields[1][1][0], bin_yields[2][1][0], bin_yields[3][1][0]])
  bin_table.append(['4', f'${bin_yields[1][0][2]}$', f'{significances[0]:.3f}+-{significance_errs[0]:.3f}', f'{bin_yields[1][0][1]:.3f}', f'{bin_yields[2][0][1]:.3f}', f'{bin_yields[3][0][1]:.3f}'])
  bin_table.append(['', '', 'no weights', bin_yields[1][0][0], bin_yields[2][0][0], bin_yields[3][0][0]])
  make_tex_table(bin_table, 'cccccc', f'tables/{mva_name}_bin_table.tex')

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Measures performance of MVA')
  parser.add_argument('-m','--methods', required=True, nargs="+", help='Method names')
  parser.add_argument('-n','--nbins', default=4, help='Number of MVA bins')
  args = parser.parse_args()

  signi_detail_dict = {}
  train_signi_detail_dict = {}
  roc_detail_dict = {}
  train_roc_detail_dict = {}
  bkg_mass_difference_detail_dict = {}
  sig_mass_difference_detail_dict = {}
  train_bkg_mass_difference_detail_dict = {}
  train_sig_mass_difference_detail_dict = {}
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
  mva_info_book = {\
    'standard_tmva_bdtg':['mva_output/standard_tmva_bdtg_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'y_ptmass']}],
    'standard_tmva_bdt_hig19014':['mva_output/standard_tmva_bdt_hig19014_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta']}],
    'tmva_bdt_tight_higwindow_var11':['mva_output/tmva_bdt_tight_higwindow_var11_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta']}],
    'tmva_bdtg_tight_higwindow_var11':['mva_output/tmva_bdtg_tight_higwindow_var11_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta']}],
    'xgboost_tight_higwindow_var11':['mva_output/xgboost_tight_higwindow_var11_results.root', {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta']}],
    'tmva_bdta_tight_higwindow_var12':['mva_output/tmva_bdta_tight_higwindow_var12_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'y_ptmass']}],
    'tmva_bdtg_tight_higwindow_var12':['mva_output/tmva_bdtg_tight_higwindow_var12_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'y_ptmass']}],
    'xgboost_tight_higwindow_var12':['mva_output/xgboost_tight_higwindow_var12_results.root', {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'y_ptmass']}],
    'tmva_bdta_tight_higwindow_var12cosh':['mva_output/tmva_bdta_tight_higwindow_var12cosh_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'y_ptinfo2mass2']}],
    'tmva_bdtg_tight_higwindow_var12cosh':['mva_output/tmva_bdtg_tight_higwindow_var12cosh_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'y_ptinfo2mass2']}],
    'xgboost_tight_higwindow_var12cosh':['mva_output/xgboost_tight_higwindow_var12cosh_results.root', {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'y_ptinfo2mass2']}],
    'tmva_bdt_decorr':['mva_output/tmva_bdt_decorr_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_eta', 'l1_eta', 'l2_eta']}],
    'tmva_bdt_decorr_cosThetacut':['mva_output/tmva_bdt_decorr_cosThetacut_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosTheta', 'costheta', 'phi', 'lly_ptmass', 'y_eta', 'l1_eta', 'l2_eta']}],
    'tmva_bdt_windowcut_hig19014':['mva_output/tmva_bdt_windowcut_hig19014_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'y_ptmass']}],
    'tmva_bdt_windowcut_decorr_hig19014':['mva_output/tmva_bdt_windowcut_decorr_hig19014_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva', 'yl_drmin', 'yl_drmax', 'cosThetamass3', 'costheta', 'phi', 'lly_ptmass', 'y_eta', 'l1_eta', 'l2_eta', 'l1_ptmass', 'l2_pt']}],
    'standard_xgboost':['mva_output/standard_xgboost_results.root', {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'y_ptmass']}],
    'standard_xgboost_hig19014':['mva_output/standard_xgboost_hig19014_results.root', {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta']}],
    'xgboost_decorr':['mva_output/xgboost_decorr_results.root', {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_eta', 'l1_eta', 'l2_eta']}],
    'xgboost_decorr_cosThetacut':['mva_output/xgboost_decorr_cosThetacut_results.root', {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosTheta', 'costheta', 'phi', 'lly_ptmass', 'y_eta', 'l1_eta', 'l2_eta']}],
    'xgboost_windowcut_hig19014':['mva_output/xgboost_windowcut_hig19014_results.root', {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'y_ptmass']}],
    'xgboost_windowcut_decorr_hig19014':['mva_output/xgboost_windowcut_decorr_hig19014_results.root', {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva', 'yl_drmin', 'yl_drmax', 'cosThetamass3', 'costheta', 'phi', 'lly_ptmass', 'y_eta', 'l1_eta', 'l2_eta', 'l1_ptmass', 'l2_pt']}],
    'tmva_bdt_nowindowcut':['mva_output/tmva_bdt_nowindowcut_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta']}],
    'xgboost_nowindowcut':['mva_output/xgboost_nowindowcut_results.root', {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta']}],
    'tmva_bdt_windowcut':['mva_output/tmva_bdt_windowcut_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta']}],
    'xgboost_windowcut':['mva_output/xgboost_windowcut_results.root', {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta']}],
    'tmva_bdt_yptmass2':['mva_output/tmva_bdt_yptmass2_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta']}],
    'xgboost_yptmass2':['mva_output/xgboost_yptmass2_results.root', {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta']}],
    'tmva_bdt_decorr_min19':['mva_output/tmva_bdt_decorr_min19_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'ht']}],
    'xgboost_decorr_min19':['mva_output/xgboost_decorr_min19_results.root', {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'ht']}],
    'tmva_bdt_decorr_min19bad':['mva_output/tmva_bdt_decorr_min19bad_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'lly_ptt', 'ht']}],
    'xgboost_decorr_min19bad':['mva_output/xgboost_decorr_min19bad_results.root', {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'lly_ptt', 'ht']}],
    'tmva_bdt_min14_ypt':['mva_output/tmva_bdt_min14_ypt_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'ht', 'y_ptinfo2mass2']}],
    'xgboost_min14_ypt':['mva_output/xgboost_min14_ypt_results.root', {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'ht', 'y_ptinfo2mass2']}],
    'tmva_bdt_min13':['mva_output/tmva_bdt_min13_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']}],
    'xgboost_min13':['mva_output/xgboost_min13_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']}],
    'tmva_bdtg_min13':['mva_output/tmva_bdtg_min13_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']}],
    'tmva_bdt_decorr18':['mva_output/tmva_bdt_decorr18_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'l1_ptmass', 'l2_pt', 'leplep_flavor', 'leplep_m', 'leplep_eta', 'lly_ptt', 'lly_eta']}],
    'xgboost_decorr18':['mva_output/xgboost_decorr18_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3],            'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'l1_ptmass', 'l2_pt', 'leplep_flavor', 'leplep_m', 'leplep_eta', 'lly_ptt', 'lly_eta']}],
    'tmva_bdt_min13_ypt':['mva_output/tmva_bdt_min13_ypt_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'y_ptinfo2mass2']}],
    'xgboost_min13_ypt':['mva_output/xgboost_min13_ypt_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'y_ptinfo2mass2']}],
    'tmva_bdtg_min13_ypt':['mva_output/tmva_bdtg_min13_ypt_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'y_ptinfo2mass2']}],
    'tmva_bdt19':['mva_output/tmva_bdt19_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'l1_ptmass', 'l2_pt', 'leplep_flavor', 'leplep_m', 'leplep_eta', 'lly_ptt', 'lly_eta','y_ptinfo2mass2']}],
    'xgboost19':['mva_output/xgboost19_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3],            'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'l1_ptmass', 'l2_pt', 'leplep_flavor', 'leplep_m', 'leplep_eta', 'lly_ptt', 'lly_eta','y_ptinfo2mass2']}],
    'tmva_bdtg_yptmass':['mva_output/tmva_bdtg_yptmass_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'y_ptmass']}],
    #'etn14featdecorr2drop15':['trash/etn14featdecorr2drop15_extdisco10_2024_05_05_03-34-31.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'y_ptinfo2mass2']}],
    'etn14featdecorr2drop15extdisco10':['mva_output/etn14featdecorr2drop15_extdisco10.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'y_ptinfo2mass2']}],
    'etn14featdecorr2drop15extdisco15':['mva_output/etn14featdecorr2drop15_extdisco15.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'y_ptinfo2mass2']}],
    'etn14featdecorr2drop15extdisco20':['mva_output/etn14featdecorr2drop15_extdisco20.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'y_ptinfo2mass2']}],
    'etn14featdecorr2drop15extdisco30':['mva_output/etn14featdecorr2drop15_extdisco30.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'y_ptinfo2mass2']}],
    'tmva_bdta_tight_higwindow_var13':['mva_output/tmva_bdta_tight_higwindow_var13_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']}],
    'tmva_bdtg_tight_higwindow_var13':['mva_output/tmva_bdtg_tight_higwindow_var13_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']}],
    'xgboost_tight_higwindow_var13':['mva_output/xgboost_tight_higwindow_var13_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']}],
    'tmva_bdta_tight0j_higwindow_var13':['mva_output/tmva_bdta_tight0j_higwindow_var13_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']}],
    'tmva_bdta_tight1j_higwindow_var13':['mva_output/tmva_bdta_tight1j_higwindow_var13_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']}],
    'tmva_bdta_ggfvbf_tight_higwindow_var13':['mva_output/tmva_bdta_ggfvbf_tight_higwindow_var13_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']}],
    'tmva_bdta_ggfvbf_tight0j_higwindow_var13':['mva_output/tmva_bdta_ggfvbf_tight0j_higwindow_var13_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']}],
    'tmva_bdta_ggfvbf_tight1j_higwindow_var13':['mva_output/tmva_bdta_ggfvbf_tight1j_higwindow_var13_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']}],
    'tmva_bdta_tight1j_higwindow_var19':['mva_output/tmva_bdta_tight1j_higwindow_var19_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'j1_pt', 'j1_eta', 'j1_phi', 'llyj_dphi', 'yj1_deta', 'yj1_dr', 'llyj1_ptbal']}],
    'tmva_bdta_ggfvbf_tight1j_higwindow_var19':['mva_output/tmva_bdta_ggfvbf_tight1j_higwindow_var19_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'j1_pt', 'j1_eta', 'j1_phi', 'llyj_dphi', 'yj1_deta', 'yj1_dr', 'llyj1_ptbal']}],
    'tmva_bdta_ggfvbf_tight1j_higwindow_var15':['mva_output/tmva_bdta_ggfvbf_tight1j_higwindow_var15_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']}],
    'tmva_bdta_ggfvbf_tight_higwindow_var15':['mva_output/tmva_bdta_ggfvbf_tight_higwindow_var15_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']}],
    'tmva_bdta_tight_higwindow_var12':['mva_output/tmva_bdta_tight_higwindow_var12_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']}],
    'tmva_bdta_tight_higwindow_var10':['mva_output/tmva_bdta_tight_higwindow_var10_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva', 'yl_drmin', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']}],
    'tmva_bdta_tight_higwindow_var7':['mva_output/tmva_bdta_tight_higwindow_var7_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva', 'yl_drmin', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'lly_m']}],
    'tmva_bdta_tight_higwindow_var6':['mva_output/tmva_bdta_tight_higwindow_var6_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva', 'yl_drmin', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'lly_m']}],
    'tmva_bdta_tight_higwindow_var5':['mva_output/tmva_bdta_tight_higwindow_var5_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva', 'yl_drmin', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'lly_m']}],
    'tmva_bdta_tight_higwindow_var3':['mva_output/tmva_bdta_tight_higwindow_var5_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva', 'yl_drmin', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'lly_m']}],
    'tmva_bdta_tight_higwindow_var14':['mva_output/tmva_bdta_tight_higwindow_var14_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'npv']}],
    'xgboost_tight_higwindow_var14':['mva_output/xgboost_tight_higwindow_var14_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'npv']}],
    'xgboost_tight0j_higwindow_var13':['mva_output/xgboost_tight0j_higwindow_var13_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']}],
    'xgboost_tight1j_higwindow_var13':['mva_output/xgboost_tight1j_higwindow_var13_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']}],
    'xgboost_tight1j_higwindow_var19':['mva_output/xgboost_tight1j_higwindow_var19_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'j1_pt', 'j1_eta', 'j1_phi', 'llyj_dphi', 'yj1_deta', 'yj1_dr', 'llyj1_ptbal']}],
    'xgboost_ggfvbf_tight_higwindow_var13':['mva_output/xgboost_ggfvbf_tight_higwindow_var13_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']}],
    'xgboost_ggfvbf_tight0j_higwindow_var13':['mva_output/xgboost_ggfvbf_tight0j_higwindow_var13_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']}],
    'xgboost_ggfvbf_tight1j_higwindow_var13':['mva_output/xgboost_ggfvbf_tight1j_higwindow_var13_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']}],
    'xgboost_ggfvbf_tight_higwindow_var19':['mva_output/xgboost_ggfvbf_tight_higwindow_var19_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'j1_pt', 'j1_eta', 'j1_phi', 'llyj_dphi', 'yj1_deta', 'yj1_dr', 'llyj1_ptbal']}],
    'xgboost_ggfvbf_tight0j_higwindow_var19':['mva_output/xgboost_ggfvbf_tight0j_higwindow_var19_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'j1_pt', 'j1_eta', 'j1_phi', 'llyj_dphi', 'yj1_deta', 'yj1_dr', 'llyj1_ptbal']}],
    'xgboost_ggfvbf_tight1j_higwindow_var19':['mva_output/xgboost_ggfvbf_tight1j_higwindow_var19_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'j1_pt', 'j1_eta', 'j1_phi', 'llyj_dphi', 'yj1_deta', 'yj1_dr', 'llyj1_ptbal']}],
    'xgboost_ggfvbf_tight1j_higwindow_var15':['mva_output/xgboost_ggfvbf_tight1j_higwindow_var15_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']}],
    'xgboost_ggfvbf_tight_higwindow_var15':['mva_output/xgboost_ggfvbf_tight_higwindow_var15_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']}],
    'xgboost_wgt_ggfvbf_tight_higwindow_var15':  ['mva_output/xgboost_wgt_ggfvbf_tight_higwindow_var15_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']}],
    'xgboost_wgt_ggfvbf_tight_higwindow_var13':  ['mva_output/xgboost_wgt_ggfvbf_tight_higwindow_var13_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']}],
    'xgboost_wgt_ggfvbf_tight0j_higwindow_var13':['mva_output/xgboost_wgt_ggfvbf_tight0j_higwindow_var13_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']}],
    'xgboost_wgt_ggfvbf_tight1j_higwindow_var15':['mva_output/xgboost_wgt_ggfvbf_tight1j_higwindow_var15_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']}],
    'xgboost_wgt11_ggfvbf_tight_higwindow_var15':  ['mva_output/xgboost_wgt11_ggfvbf_tight_higwindow_var15_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']}],
    'xgboost_wgt11_ggfvbf_tight_higwindow_var13':  ['mva_output/xgboost_wgt11_ggfvbf_tight_higwindow_var13_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']}],
    'xgboost_wgt11_ggfvbf_tight0j_higwindow_var13':['mva_output/xgboost_wgt11_ggfvbf_tight0j_higwindow_var13_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']}],
    'xgboost_wgt11_ggfvbf_tight1j_higwindow_var15':['mva_output/xgboost_wgt11_ggfvbf_tight1j_higwindow_var15_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']}],
    'xgboost_wgt111_ggfvbf_tight_higwindow_var15':  ['mva_output/xgboost_wgt111_ggfvbf_tight_higwindow_var15_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']}],
    'xgboost_wgt111_ggfvbf_tight_higwindow_var13':  ['mva_output/xgboost_wgt111_ggfvbf_tight_higwindow_var13_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']}],
    'xgboost_wgt111_ggfvbf_tight0j_higwindow_var13':['mva_output/xgboost_wgt111_ggfvbf_tight0j_higwindow_var13_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']}],
    'xgboost_wgt111_ggfvbf_tight1j_higwindow_var15':['mva_output/xgboost_wgt111_ggfvbf_tight1j_higwindow_var15_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']}],
    'etndr0p1var19_bcebce_ggfvbf_tight_higwindow': ['mva_output/etndr0p1var19_bcebce_ggfvbf_tight_higwindow.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'j1_pt', 'j1_eta', 'j1_phi', 'llyj_dphi', 'yj1_deta', 'yj1_dr', 'llyj1_ptbal']}],
    'etndr0p1var15_bcebce_ggfvbf_tight_higwindow': ['mva_output/etndr0p1var15_bcebce_ggfvbf_tight_higwindow.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']}],
    'etndr0p1var13_bcebce_ggfvbf_tight_higwindow': ['mva_output/etndr0p1var13_bcebce_ggfvbf_tight_higwindow.root',   {'y':'y', 'yhat':'yhat', '"1"': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']}],
    'xgboost_wgtmass_ggfvbf_tight_higwindow_var15':  ['mva_output/xgboost_wgtmass_ggfvbf_tight_higwindow_var15_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']}],
    'xgboost_wgtmass11_ggfvbf_tight_higwindow_var15':  ['mva_output/xgboost_wgtmass11_ggfvbf_tight_higwindow_var15_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']}],
    'xgboost_wgtres11_ggfvbf_tight_higwindow_var15':  ['mva_output/xgboost_wgtres11_ggfvbf_tight_higwindow_var15_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']}],
    'xgboost_wgtsigres11_ggfvbf_tight_higwindow_var15':  ['mva_output/xgboost_wgtsigres11_ggfvbf_tight_higwindow_var15_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']}],
    'xgboost_wgtsigres211_ggfvbf_tight_higwindow_var15':  ['mva_output/xgboost_wgtsigres211_ggfvbf_tight_higwindow_var15_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5],              'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']}],
    'tmva_bdta_wgt11_ggfvbf_tight_higwindow_var15':['mva_output/tmva_bdta_wgt11_ggfvbf_tight_higwindow_var15_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']}],
    'tmva_bdta_wgt_ggfvbf_tight_higwindow_var15':['mva_output/tmva_bdta_wgt_ggfvbf_tight_higwindow_var15_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']}],
    'tmva_bdta_wgteq1_ggfvbf_tight_higwindow_var15':['mva_output/tmva_bdta_wgteq1_ggfvbf_tight_higwindow_var15_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']}],
    'tmva_bdta_wgtsigres11_ggfvbf_tight_higwindow_var15':['mva_output/tmva_bdta_wgtsigres11_ggfvbf_tight_higwindow_var15_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']}],
  }

  mva_nbins = int(args.nbins)

  mva_info = {}
  for method in args.methods:
    if method in mva_info_book: mva_info[method] = mva_info_book[method]
    else:
      print(f'[Error] No method called {method}')
      sys.exit()

  if f'-I{os.environ["WORK_DIR"]}/root_scripts' not in ROOT.gSystem.GetIncludePath():
    ROOT.gSystem.AddIncludePath(f'-I{os.environ["WORK_DIR"]}/root_scripts')
  ROOT.gInterpreter.ProcessLine('.L evaluate_mva.C++')


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
    binned_significance, binned_significance_err, binned_signi_detail = measure_tools.calculate_binned_significance(mva_filename, 'eval_tree_baseline', branches, detail_output=True, mva_nbins=mva_nbins, throw_away_bin=0)
    print(f'[{mva_name}] Significance {mva_nbins} bins: {binned_significance:.3f} +- {binned_significance_err:.3f}')
    train_binned_significance, train_binned_significance_err, train_binned_signi_detail = measure_tools.calculate_binned_significance(mva_filename, 'train_tree_baseline', branches, detail_output=True, mva_nbins=mva_nbins, throw_away_bin=0)
    print(f'[{mva_name}] Significance {mva_nbins} bins (train): {train_binned_significance:.3f} +- {train_binned_significance_err:.3f}')
    auc_sci, auc_sci_detail = measure_tools.calculate_auc_sci(mva_filename, 'eval_tree', branches, detail_output=True)
    print(f'[{mva_name}] AUC: {auc_sci[0]*100:.1f}% CSI: {auc_sci[1]:.2f}')
    train_auc_sci, train_auc_sci_detail = measure_tools.calculate_auc_sci(mva_filename, 'train_tree', branches, detail_output=True)
    print(f'[{mva_name}] AUC (train): {train_auc_sci[0]*100:.1f}% CSI (train): {train_auc_sci[1]:.2f}')
    # Calculation for plotting
    sig_mass_shape_difference, sig_mass_difference_detail = measure_tools.calculate_mass_shape_difference(mva_filename, 'eval_tree_baseline', branches, y_value=1,detail_output=True)
    bkg_feature_shape_difference = measure_tools.find_feature_shape_difference(mva_filename, 'eval_tree_baseline', branches, y_value=0)
    sig_feature_shape_difference = measure_tools.find_feature_shape_difference(mva_filename, 'eval_tree_baseline', branches, y_value=1)
    bkg_feature_shape_difference_in_bin = measure_tools.find_feature_shape_difference_in_bins(mva_filename, 'eval_tree_baseline', branches, mva_bins=binned_signi_detail[1][3][1], y_value=0)
    sig_feature_shape_difference_in_bin = measure_tools.find_feature_shape_difference_in_bins(mva_filename, 'eval_tree_baseline', branches, mva_bins=train_binned_signi_detail[1][3][1], y_value=1)
    train_bkg_mass_shape_difference, train_bkg_mass_difference_detail = measure_tools.calculate_mass_shape_difference(mva_filename, 'train_tree_baseline', branches, detail_output=True)
    train_sig_mass_shape_difference, train_sig_mass_difference_detail = measure_tools.calculate_mass_shape_difference(mva_filename, 'train_tree_baseline', branches, y_value=1,detail_output=True)

    # Direct plotting
    measure_tools.plot_feature_shapes(mva_filename, 'train_tree_baseline', branches['observable'], branches["x"], branches['weight'], f'{branches["y"]}==1', f'{branches["y"]}==0', f'{mva_name}_bkg')
    #measure_tools.plot_feature_shapes(mva_filename, 'train_tree_baseline', branches['observable'], branches["x"], branches['weight'], f'{branches["y"]}==1', f'{branches["y"]}==1', f'{mva_name}_sig')

    # Directly making mva table from binned_signi_detail
    if mva_nbins == 4: make_mva_table(binned_signi_detail)

    # Collect detail for plots
    signi_detail_dict[mva_name] = signi_detail
    train_signi_detail_dict[mva_name] = train_signi_detail
    roc_detail_dict[mva_name] = roc_detail
    train_roc_detail_dict[mva_name] = train_roc_detail
    bkg_mass_difference_detail_dict[mva_name] = bkg_mass_difference_detail
    sig_mass_difference_detail_dict[mva_name] = sig_mass_difference_detail
    train_bkg_mass_difference_detail_dict[mva_name] = train_bkg_mass_difference_detail
    train_sig_mass_difference_detail_dict[mva_name] = train_sig_mass_difference_detail
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
  measure_tools.draw_mass_shape_difference(train_bkg_mass_difference_detail_dict,name_tag='train_bkg')
  measure_tools.draw_mass_shape_difference(train_sig_mass_difference_detail_dict,name_tag='train_sig')
  measure_tools.draw_overtraining(overtrain_detail_dict)
  measure_tools.draw_binned_signi_detail(binned_signi_detail_dict)
  measure_tools.draw_binned_signi_detail(train_binned_signi_detail_dict,name_tag='train')
  measure_tools.draw_binned_signi_detail_train_eval(binned_signi_detail_dict, train_binned_signi_detail_dict)
  measure_tools.draw_auc_sci_detail(auc_sci_detail_dict)
  measure_tools.draw_sci_detail_train_eval(auc_sci_detail_dict, train_auc_sci_detail_dict)
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
    bin_plots = ''
    for ibin in range(mva_nbins):
      bin_plots += f'plots/{mva_name}_binned_mva_mlly_bin{ibin}_per_sample.pdf '
    command = f'pdf_combine.py -f -i {bin_plots} plots/{mva_name}_binned_mva_per_sample.pdf plots/{mva_name}_bkg_hist_mva.pdf plots/{mva_name}_sig_hist_mva.pdf plots/{mva_name}_signal_resolution.pdf'
    command += f' -x 4 -y 2 -o plots/summary_{mva_name}_mass_shape.pdf'
    os.system(command)
