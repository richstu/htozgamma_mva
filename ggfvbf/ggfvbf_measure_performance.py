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

def make_mva_table(binned_signi_detail, mva_nbins, sample_ids):
  mva_nbins, binned_signi_detail_list, sig_hist, bkg_hist, sig_obs_hists, bkg_obs_hists, mva_hists, obs_hists, sig_ids, bkg_ids = binned_signi_detail
  nbins, bin_edges, significance, significance_err, significances, significance_errs, signal_yield, background_yield = binned_signi_detail_list[mva_nbins-1]
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
  for sample_id in sample_ids:
    sample_cut = f'{branches["sample_id"]}=={sample_id}'
    if sample_id not in bin_yields: bin_yields[sample_id] = {}
    # Loop over each bin
    for ibin in range(nbins):
      if ibin==0: bin_cut = f'{branches["yhat"]}<{bin_edges[0]}'
      elif ibin==nbins-1: bin_cut = f'{branches["yhat"]}>={bin_edges[-1]}'
      else: bin_cut = f'{branches["yhat"]}>={bin_edges[ibin-1]}&&{branches["yhat"]}<{bin_edges[ibin]}'
      # Get yield
      chain.Draw(f'{branches["observable"]}>>{hist_name}', f'({significance_window_cut}&&{sample_cut}&&{bin_cut})*{branches["weight"]}', 'goff')
      sample_yield_err = ctypes.c_double()
      sample_yield = hist.IntegralAndError(0, hist.GetNbinsX()+1, sample_yield_err)
      sample_yield_err = sample_yield_err.value
      #bin_yields[sample_id][ibin] = [hist.GetEntries(), hist.GetSum(), bin_cut.replace('&&',r'\&')]
      bin_yields[sample_id][ibin] = [hist.GetEntries(), sample_yield, sample_yield_err, bin_cut.replace('&&',r'\&')]
  # Make tex table using yields
  bin_table = []
  bin_table.append(['MVA bin', 'MVA cut', 'Significance'])
  bin_table[-1].extend(list(sample_ids.values()))
  ncolumns = len(list(sample_ids.values())) + 3
  for imva in range(mva_nbins):
    bin_index = mva_nbins-imva-1
    row_list = [f'{imva+1}', f'${bin_yields[1][bin_index][3]}$', f'{significances[bin_index]:.3f}+-{significance_errs[bin_index]:.3f}']
    for isample in sample_ids:
      row_list.append(f'{bin_yields[isample][bin_index][1]:.3f}+-{bin_yields[isample][bin_index][2]:.3f}')
    bin_table.append(row_list)
    row_list = ['', '', 'no weights']
    for isample in sample_ids:
      row_list.append(bin_yields[isample][bin_index][0])
    bin_table.append(row_list)
  make_tex_table(bin_table, 'c'*ncolumns, f'tables/{mva_name}_bin_table.tex')

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
    'etndr0p1var19_bce_ggfvbf_tight_012j_higwindow_2j': ['mva_output/etndr0p1var26_bce_ggfvbf_tight_012j_higwindow_2j.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'j1_pt', 'j1_eta', 'llyj_dphi', 'yj1_deta', 'yj1_dr', 'llyj1_ptbal', 'jj_deta', 'jj_dphi', 'yj2_dr', 'llyjj_dphi', 'j2_pt', 'llyjj_ptbal', 'yjj_zep']}],
    'etndr0p1var19_bce_ggfvbf_tight_012j_higwindow_01j': ['mva_output/etndr0p1var26_bce_ggfvbf_tight_012j_higwindow_01j.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'j1_pt', 'j1_eta', 'llyj_dphi', 'yj1_deta', 'yj1_dr', 'llyj1_ptbal', 'jj_deta', 'jj_dphi', 'yj2_dr', 'llyjj_dphi', 'j2_pt', 'llyjj_ptbal', 'yjj_zep']}],
    'etndr0p1var19_bce_ggfvbf_tight_012j_higwindow_1j': ['mva_output/etndr0p1var26_bce_ggfvbf_tight_012j_higwindow_1j.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'j1_pt', 'j1_eta', 'llyj_dphi', 'yj1_deta', 'yj1_dr', 'llyj1_ptbal', 'jj_deta', 'jj_dphi', 'yj2_dr', 'llyjj_dphi', 'j2_pt', 'llyjj_ptbal', 'yjj_zep']}],
    'etndr0p1var19_bce_ggfvbf_tight_012j_higwindow_0j': ['mva_output/etndr0p1var26_bce_ggfvbf_tight_012j_higwindow_0j.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'j1_pt', 'j1_eta', 'llyj_dphi', 'yj1_deta', 'yj1_dr', 'llyj1_ptbal', 'jj_deta', 'jj_dphi', 'yj2_dr', 'llyjj_dphi', 'j2_pt', 'llyjj_ptbal', 'yjj_zep']}],
    'etndr0p1var19_bce_ggfvbf_tight_012j_higwindow': ['mva_output/etndr0p1var26_bce_ggfvbf_tight_012j_higwindow.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'j1_pt', 'j1_eta', 'llyj_dphi', 'yj1_deta', 'yj1_dr', 'llyj1_ptbal', 'jj_deta', 'jj_dphi', 'yj2_dr', 'llyjj_dphi', 'j2_pt', 'llyjj_ptbal', 'yjj_zep']}],
    'emtndr0p1var26_bce_ggfvbf_tight_012j_higwindow': ['mva_output/emtndr0p1var26_bce_ggfvbf_tight_012j_higwindow.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'j1_pt', 'j1_eta', 'llyj_dphi', 'yj1_deta', 'yj1_dr', 'llyj1_ptbal', 'jj_deta', 'jj_dphi', 'yj2_dr', 'llyjj_dphi', 'j2_pt', 'llyjj_ptbal', 'yjj_zep']}],
    'emtndr0p1var26_bce_ggfvbf_tight_012j_higwindow_2j': ['mva_output/emtndr0p1var26_bce_ggfvbf_tight_012j_higwindow_2j.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'j1_pt', 'j1_eta', 'llyj_dphi', 'yj1_deta', 'yj1_dr', 'llyj1_ptbal', 'jj_deta', 'jj_dphi', 'yj2_dr', 'llyjj_dphi', 'j2_pt', 'llyjj_ptbal', 'yjj_zep']}],

    'etfndr0p1var26_bce_ggfvbf_tight_012j_higwindow': ['mva_output/etfndr0p1var26_bce_ggfvbf_tight_012j_higwindow.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'j1_pt', 'j1_eta', 'llyj_dphi', 'yj1_deta', 'yj1_dr', 'llyj1_ptbal', 'jj_deta', 'jj_dphi', 'yj2_dr', 'llyjj_dphi', 'j2_pt', 'llyjj_ptbal', 'yjj_zep']}],
    'etfndr0p1var26_ft_bce_ggfvbf_tight_2j_higwindow': ['mva_output/etfndr0p1var26_ft_bce_ggfvbf_tight_2j_higwindow.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'j1_pt', 'j1_eta', 'llyj_dphi', 'yj1_deta', 'yj1_dr', 'llyj1_ptbal', 'jj_deta', 'jj_dphi', 'yj2_dr', 'llyjj_dphi', 'j2_pt', 'llyjj_ptbal', 'yjj_zep']}],
    'etfndr0p1var26_ft_bce_ggfvbf_tight_1j_higwindow': ['mva_output/etfndr0p1var26_ft_bce_ggfvbf_tight_1j_higwindow.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'j1_pt', 'j1_eta', 'llyj_dphi', 'yj1_deta', 'yj1_dr', 'llyj1_ptbal', 'jj_deta', 'jj_dphi', 'yj2_dr', 'llyjj_dphi', 'j2_pt', 'llyjj_ptbal', 'yjj_zep']}],
    'etfndr0p1var26_ft_bce_ggfvbf_tight_0j_higwindow': ['mva_output/etfndr0p1var26_ft_bce_ggfvbf_tight_0j_higwindow.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'j1_pt', 'j1_eta', 'llyj_dphi', 'yj1_deta', 'yj1_dr', 'llyj1_ptbal', 'jj_deta', 'jj_dphi', 'yj2_dr', 'llyjj_dphi', 'j2_pt', 'llyjj_ptbal', 'yjj_zep']}],

    'etfndr0p1var26_ext_ggfvbf_tight_012j_higwindow': ['mva_output/etfndr0p1var26_ext_ggfvbf_tight_012j_higwindow.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'j1_pt', 'j1_eta', 'llyj_dphi', 'yj1_deta', 'yj1_dr', 'llyj1_ptbal', 'jj_deta', 'jj_dphi', 'yj2_dr', 'llyjj_dphi', 'j2_pt', 'llyjj_ptbal', 'yjj_zep']}],
    'etfndr0p1var26_ft_ext_ggfvbf_tight_2j_higwindow': ['mva_output/etfndr0p1var26_ft_ext_ggfvbf_tight_2j_higwindow.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'j1_pt', 'j1_eta', 'llyj_dphi', 'yj1_deta', 'yj1_dr', 'llyj1_ptbal', 'jj_deta', 'jj_dphi', 'yj2_dr', 'llyjj_dphi', 'j2_pt', 'llyjj_ptbal', 'yjj_zep']}],

    'etfndr0p1var26_bce_ggfvbf_tight_2j_higwindow': ['mva_output/etfndr0p1var26_bce_ggfvbf_tight_2j_higwindow.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'j1_pt', 'j1_eta', 'llyj_dphi', 'yj1_deta', 'yj1_dr', 'llyj1_ptbal', 'jj_deta', 'jj_dphi', 'yj2_dr', 'llyjj_dphi', 'j2_pt', 'llyjj_ptbal', 'yjj_zep']}],
    'etfndr0p1var26_ext_ggfvbf_tight_2j_higwindow': ['mva_output/etfndr0p1var26_ext_ggfvbf_tight_2j_higwindow.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'j1_pt', 'j1_eta', 'llyj_dphi', 'yj1_deta', 'yj1_dr', 'llyj1_ptbal', 'jj_deta', 'jj_dphi', 'yj2_dr', 'llyjj_dphi', 'j2_pt', 'llyjj_ptbal', 'yjj_zep']}],
    'etfndr0p1var20_ext_ggfvbf_tight_2j_higwindow': ['mva_output/etfndr0p1var20_ext_ggfvbf_tight_2j_higwindow.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva','yl_drmin','yl_drmax','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'lly_ptt', 'jj_deta', 'jj_dphi', 'yj1_dr', 'yj2_dr', 'llyjj_dphi', 'j1_pt', 'j2_pt', 'llyjj_ptbal', 'yjj_zep']}],

    'xgboost_tight_2j_higwindow_var22': ['mva_output/xgboost_tight_2j_higwindow_var22_results.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva','yl_drmin','yl_drmax','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'lly_ptt', 'jj_deta', 'jj_dphi', 'yj1_dr', 'yj2_dr', 'llyjj_dphi', 'j1_pt', 'j2_pt', 'llyjj_ptbal', 'yjj_zep']}],
    'tmva_bdta_wgt11_ggfvbf_tight_2j_higwindow_var20': ['mva_output/tmva_bdta_wgt11_ggfvbf_tight_2j_higwindow_var20_results.root',   {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva','yl_drmin','yl_drmax','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'lly_ptt', 'jj_deta', 'jj_dphi', 'yj1_dr', 'yj2_dr', 'llyjj_dphi', 'j1_pt', 'j2_pt', 'llyjj_ptbal', 'yjj_zep']}],

    'emtndr0p1var20_gnll_ggfvbf_tight_2j_higwindow': ['mva_output/emtndr0p1var20_gnll_ggfvbf_tight_2j_higwindow_convert.root',   {'y':'y', 'yhat':'yhat_single1', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva','yl_drmin','yl_drmax','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'lly_ptt', 'jj_deta', 'jj_dphi', 'yj1_dr', 'yj2_dr', 'llyjj_dphi', 'j1_pt', 'j2_pt', 'llyjj_ptbal', 'yjj_zep']}],
    'emtndr0p1var20_gnll_ggfvbf_tight_2j_higwindow_0': ['mva_output/emtndr0p1var20_gnll_ggfvbf_tight_2j_higwindow_convert.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva','yl_drmin','yl_drmax','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'lly_ptt', 'jj_deta', 'jj_dphi', 'yj1_dr', 'yj2_dr', 'llyjj_dphi', 'j1_pt', 'j2_pt', 'llyjj_ptbal', 'yjj_zep']}],
    'emtndr0p1var20_gnll_ggfvbf_tight_2j_higwindow_2': ['mva_output/emtndr0p1var20_gnll_ggfvbf_tight_2j_higwindow_convert.root',   {'y':'y', 'yhat':'yhat_single0', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva','yl_drmin','yl_drmax','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'lly_ptt', 'jj_deta', 'jj_dphi', 'yj1_dr', 'yj2_dr', 'llyjj_dphi', 'j1_pt', 'j2_pt', 'llyjj_ptbal', 'yjj_zep']}],

    'emtfndr0p1var20_gnll_ggfvbf_tight_2j_higwindow': ['mva_output/emtfndr0p1var20_gnll_ggfvbf_tight_2j_higwindow_convert.root',   {'y':'y', 'yhat':'yhat_single1', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva','yl_drmin','yl_drmax','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'lly_ptt', 'jj_deta', 'jj_dphi', 'yj1_dr', 'yj2_dr', 'llyjj_dphi', 'j1_pt', 'j2_pt', 'llyjj_ptbal', 'yjj_zep']}],
    'emtfndr0p1var20_gnll_ggfvbf_tight_2j_higwindow_0': ['mva_output/emtfndr0p1var20_gnll_ggfvbf_tight_2j_higwindow_convert.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva','yl_drmin','yl_drmax','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'lly_ptt', 'jj_deta', 'jj_dphi', 'yj1_dr', 'yj2_dr', 'llyjj_dphi', 'j1_pt', 'j2_pt', 'llyjj_ptbal', 'yjj_zep']}],

    'emtfndr0p1var26_gnll_ggfvbf_tight_2j_higwindow': ['mva_output/emtfndr0p1var26_gnll_ggfvbf_tight_2j_higwindow_convert.root',   {'y':'y', 'yhat':'yhat_single1', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva','yl_drmin','yl_drmax','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'lly_ptt', 'jj_deta', 'jj_dphi', 'yj1_dr', 'yj2_dr', 'llyjj_dphi', 'j1_pt', 'j2_pt', 'llyjj_ptbal', 'yjj_zep']}],
    'emtfndr0p1var26_gnll_ggfvbf_tight_2j_higwindow_0': ['mva_output/emtfndr0p1var26_gnll_ggfvbf_tight_2j_higwindow_convert.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva','yl_drmin','yl_drmax','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'lly_ptt', 'jj_deta', 'jj_dphi', 'yj1_dr', 'yj2_dr', 'llyjj_dphi', 'j1_pt', 'j2_pt', 'llyjj_ptbal', 'yjj_zep']}],

    'emtfndr0p2var20_gnll_ggfvbf_tight_2j_higwindow': ['mva_output/emtfndr0p2var20_gnll_ggfvbf_tight_2j_higwindow_convert.root',   {'y':'y', 'yhat':'yhat_single1', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva','yl_drmin','yl_drmax','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'lly_ptt', 'jj_deta', 'jj_dphi', 'yj1_dr', 'yj2_dr', 'llyjj_dphi', 'j1_pt', 'j2_pt', 'llyjj_ptbal', 'yjj_zep']}],
    'emtfndr0p2var20_gnll_ggfvbf_tight_2j_higwindow_0': ['mva_output/emtfndr0p2var20_gnll_ggfvbf_tight_2j_higwindow_convert.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva','yl_drmin','yl_drmax','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'lly_ptt', 'jj_deta', 'jj_dphi', 'yj1_dr', 'yj2_dr', 'llyjj_dphi', 'j1_pt', 'j2_pt', 'llyjj_ptbal', 'yjj_zep']}],

    'eutfndr0p1var20_nll_ggfvbf_tight_2j_higwindow': ['mva_output/eutfndr0p1var20_nll_ggfvbf_tight_2j_higwindow.root',   {'y':'y', 'yhat':'yhat', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva','yl_drmin','yl_drmax','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'lly_ptt', 'jj_deta', 'jj_dphi', 'yj1_dr', 'yj2_dr', 'llyjj_dphi', 'j1_pt', 'j2_pt', 'llyjj_ptbal', 'yjj_zep']}],
    'tmva_bdta_ggfvbf_tight_01j_higwindow_var12': ['mva_output/tmva_bdta_ggfvbf_tight_01j_higwindow_var12_results.root',   {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva','yl_drmin','yl_drmax','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'leplep_m']}],

    'tmva_bdta_ggfvbf_tight_01j_higwindow_var11': ['mva_output/tmva_bdta_ggfvbf_tight_01j_higwindow_var11_results.root',   {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva', 'yl_drmin', 'yl_drmax', 'cosTheta', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta']}],
    'tmva_bdta_ggfvbf_tight_2j_higwindow_var21': ['mva_output/tmva_bdta_ggfvbf_tight_2j_higwindow_var21_results.root',   {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2,3], 'sig_ids':[4,5], 'x':['y_mva','yl_drmin','yl_drmax','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'lly_ptmass', 'lly_ptt', 'jj_deta', 'jj_dphi', 'yj1_dr', 'yj2_dr', 'llyjj_dphi', 'j1_pt', 'j2_pt', 'llyjj_ptbal', 'yjj_zep']}],

    'tmva_bdta_peking_higwindow_var11': ['mva_output/tmva_bdta_peking_higwindow_var11_results.root',   {'y':'classID', 'yhat':'BDT', 'weight': 'weight', 'observable':'H_mass', 'sample_id':'sampleID', 'bkg_ids':[1], 'sig_ids':[2], 'x':['gamma_mvaID', 'l2g_deltaR', 'l1g_deltaR', 'Z_cos_theta', 'lep_cos_theta', 'lep_phi', 'H_ptt', 'gamma_ptRelErr', 'gamma_eta', 'Z_lead_lepton_eta', 'Z_sublead_lepton_eta']}],
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
    #integrated_100_significance, integrated_100_significance_err = measure_tools.calculate_significance(mva_filename, 'eval_tree_baseline', branches, fixed_width=False, detail_output=False, nbins=100)
    #print(f'[{mva_name}] Total 100bin significance: {integrated_100_significance:.3f}+-{integrated_100_significance_err:.3f}')
    #train_integrated_100_significance, train_integrated_100_significance_err = measure_tools.calculate_significance(mva_filename, 'train_tree_baseline', branches, fixed_width=False, detail_output=False, nbins=100)
    #print(f'[{mva_name}] Total 100bin significance (train): {train_integrated_100_significance:.3f}+-{train_integrated_100_significance_err:.3f}')
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
    binned_significance_resolution, binned_significance_resolution_err = measure_tools.calculate_binned_significance_resolution(mva_filename, 'eval_tree_baseline', branches, detail_output=False, mva_nbins=mva_nbins, throw_away_bin=0)
    print(f'[{mva_name}] Significance {mva_nbins} bins resolution: {binned_significance_resolution:.3f} +- {binned_significance_resolution_err:.3f}')
    #binned_significance_resolution, binned_significance_resolution_err, binned_signi_detail= measure_tools.calculate_binned_significance_resolution(mva_filename, 'eval_tree_baseline', branches, detail_output=True, mva_nbins=mva_nbins, throw_away_bin=0)
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
    bkg_feature_shape_difference_in_bin = measure_tools.find_feature_shape_difference_in_bins(mva_filename, 'eval_tree_baseline', branches, mva_bins=binned_signi_detail[1][mva_nbins-1][1], y_value=0)
    sig_feature_shape_difference_in_bin = measure_tools.find_feature_shape_difference_in_bins(mva_filename, 'eval_tree_baseline', branches, mva_bins=train_binned_signi_detail[1][mva_nbins-1][1], y_value=1)
    train_bkg_mass_shape_difference, train_bkg_mass_difference_detail = measure_tools.calculate_mass_shape_difference(mva_filename, 'train_tree_baseline', branches, detail_output=True)
    train_sig_mass_shape_difference, train_sig_mass_difference_detail = measure_tools.calculate_mass_shape_difference(mva_filename, 'train_tree_baseline', branches, y_value=1,detail_output=True)

    # Direct plotting
    measure_tools.plot_feature_shapes(mva_filename, 'train_tree_baseline', branches['observable'], branches["x"], branches['weight'], f'{branches["y"]}==1', f'{branches["y"]}==0', f'{mva_name}_bkg')
    #measure_tools.plot_feature_shapes(mva_filename, 'train_tree_baseline', branches['observable'], branches["x"], branches['weight'], f'{branches["y"]}==1', f'{branches["y"]}==1', f'{mva_name}_sig')

    ## Directly making mva table from binned_signi_detail
    sample_ids = {1:'background', 2:'signal'}
    #sample_ids = {1:'SMZG', 2:'DYJets', 3:'VBSZG', 4:'GGF', 5:'VBF'}
    make_mva_table(binned_signi_detail, mva_nbins, sample_ids)

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
    max_plots = 15
    validation_index = 2
    input_branches = ''
    for iplot, input_branch in enumerate(mva_info[mva_name][1]['x']): 
      input_branches += f'plots/{mva_name}_bkg_8bin_hist_mva_{measure_tools.slugify(input_branch)}.pdf '
      if (iplot+1)%max_plots == 0 or iplot == len(mva_info[mva_name][1]['x'])-1:
        command = f'pdf_combine.py -f -i {input_branches} -x 5 -y 3 -o plots/summary_{mva_name}_validation_{validation_index}.pdf'
        os.system(command)
        validation_index += 1
        input_branches = ''

    input_branches = ''
    for iplot, input_branch in enumerate(mva_info[mva_name][1]['x']): 
      input_branches += f'plots/{mva_name}_bkg_4bin_hist_mva_{measure_tools.slugify(input_branch)}.pdf '
      if (iplot+1)%max_plots == 0 or iplot == len(mva_info[mva_name][1]['x'])-1:
        command = f'pdf_combine.py -f -i {input_branches} -x 5 -y 3 -o plots/summary_{mva_name}_validation_{validation_index}.pdf'
        os.system(command)
        validation_index += 1
        input_branches = ''

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
