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

def get_max_th1():
  pad_list = ROOT.gPad.GetListOfPrimitives()
  maximum = 0
  for iobj, obj in enumerate(pad_list):
    class_name = obj.ClassName()
    if "TH1" in class_name:
      t_max = obj.GetMaximum()
      if t_max>maximum or iobj == 0: maximum = t_max;
  return maximum
def set_max(maximum):
  pad_list = ROOT.gPad.GetListOfPrimitives()
  for obj in pad_list:
    class_name = obj.ClassName()
    if 'TH1' in class_name: obj.SetMaximum(maximum)
    if 'THStack' in class_name: obj.SetMaximum(maximum)
  ROOT.gPad.Modified()
  ROOT.gPad.Update()
def set_max_th1(max_fraction = 1.05):
  maximum = get_max_th1() * max_fraction
  set_max(maximum)

# Used branches: {'x', 'y', 'yhat', 'weight', 'observable'}
# display_histograms[branch] = [normalized histograms per mva bin]
def find_feature_shape_difference(root_filename, tree_name, branches, y_value=0):
  signal_fraction_edges = [0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]

  tmva_chain = ROOT.TChain(tree_name)
  tmva_chain.Add(root_filename)
  mva_threshold_edges = measure_tools.find_signal_fraction_thresholds(signal_fraction_edges, tmva_chain, branches['yhat'], branches['y'], branches['weight'])
  # Add minimum and maximum
  mva_threshold_edges.insert(0,tmva_chain.GetMinimum(branches['yhat']))
  mva_threshold_edges.append(tmva_chain.GetMaximum(branches['yhat']))

  # Storage for detail plots
  display_histograms = {}
  signal_fraction_bins = []

  # Find min, max for each feature. Use 0.05,0.95 quantiles
  x_min_max = {}
  for branch in branches['x']:
    hist = ROOT.TH1F(f"hist_{branch}",f"hist_{branch}",200,0,0)
    tmva_chain.Draw(f'{branch}>>hist_{branch}')
    branch_fractions = [0.01, 0.99]
    branch_quantiles = array.array('d', [0.]*len(branch_fractions))
    branch_fractions = array.array('d', branch_fractions)
    hist.GetQuantiles(len(branch_fractions), branch_quantiles, branch_fractions)
    x_min_max[branch] = branch_quantiles.tolist()

  #for branch in branches['x']:
  #  x_min_max[branch] = []
  #  x_min_max[branch].append(tmva_chain.GetMinimum(branch))
  #  x_min_max[branch].append(tmva_chain.GetMaximum(branch))

  for ithresh, mva_threshold in enumerate(mva_threshold_edges):
    if ithresh == 0: continue
    # Make mva window cut
    if ithresh == 1: mva_window = f'{branches["yhat"]}<{mva_threshold}'
    elif ithresh == len(mva_threshold_edges)-1: mva_window = f'{branches["yhat"]}>{mva_threshold_edges[ithresh-1]}'
    else: mva_window = f'{branches["yhat"]}<{mva_threshold}&&{branches["yhat"]}>{mva_threshold_edges[ithresh-1]}'
    # Calculate signal fraction
    if ithresh == len(mva_threshold_edges)-1: signal_fraction_bins.append(1./len(signal_fraction_edges)/2)
    else: signal_fraction_bins.append(signal_fraction_edges[ithresh-1]+1./len(signal_fraction_edges)/2)
    # Hists for display_branches
    for branch in branches['x']:
      if branch not in display_histograms: display_histograms[branch] = []
      hist = ROOT.TH1F(f"hist_{mva_threshold}_y{y_value}_{branch}",f"hist_{mva_threshold}_y{y_value}_{branch}",100, x_min_max[branch][0], x_min_max[branch][1])
      tmva_chain.Draw(f"{branch}>>hist_{mva_threshold}_y{y_value}_{branch}",f'{branches["y"]}=={y_value}&&{mva_window}',"goff")
      # Normalize histogram
      sum_weight = hist.GetSumOfWeights()
      if sum_weight != 0:
        hist.Scale(1/sum_weight)
        display_histograms[branch].append(hist)

  return signal_fraction_bins, display_histograms

# Used branches: {'x', 'y', 'yhat', 'weight', 'observable'}
# display_histograms[branch] = [normalized histograms per mva bin]
def find_feature_shape_difference_in_bins(root_filename, tree_name, branches, mva_bins, y_value=0):

  # Storage for detail plots
  display_histograms = {}
  signal_fraction_bins = []

  tmva_chain = ROOT.TChain(tree_name)
  tmva_chain.Add(root_filename)

  mva_threshold_edges = list(mva_bins)
  # Add minimum and maximum
  mva_threshold_edges.insert(0,tmva_chain.GetMinimum(branches['yhat']))
  mva_threshold_edges.append(tmva_chain.GetMaximum(branches['yhat']))

  # Find min, max for each feature. Use 0.05,0.95 quantiles
  x_min_max = {}
  for branch in branches['x']:
    hist = ROOT.TH1F(f"hist_{branch}",f"hist_{branch}",200,0,0)
    tmva_chain.Draw(f'{branch}>>hist_{branch}')
    branch_fractions = [0.01, 0.99]
    branch_quantiles = array.array('d', [0.]*len(branch_fractions))
    branch_fractions = array.array('d', branch_fractions)
    hist.GetQuantiles(len(branch_fractions), branch_quantiles, branch_fractions)
    x_min_max[branch] = branch_quantiles.tolist()

  for ithresh, mva_threshold in enumerate(mva_threshold_edges):
    if ithresh == 0: continue
    # Make mva window cut
    if ithresh == 1: mva_window = f'{branches["yhat"]}<{mva_threshold}'
    elif ithresh == len(mva_threshold_edges)-1: mva_window = f'{branches["yhat"]}>{mva_threshold_edges[ithresh-1]}'
    else: mva_window = f'{branches["yhat"]}<{mva_threshold}&&{branches["yhat"]}>{mva_threshold_edges[ithresh-1]}'
    # Calculate signal fraction
    signal_fraction_bins.append(ithresh)
    # Hists for display_branches
    for branch in branches['x']:
      if branch not in display_histograms: display_histograms[branch] = []
      hist = ROOT.TH1F(f"hist_{mva_threshold}_y{y_value}_{branch}",f"hist_{mva_threshold}_y{y_value}_{branch}",100, x_min_max[branch][0], x_min_max[branch][1])
      tmva_chain.Draw(f"{branch}>>hist_{mva_threshold}_y{y_value}_{branch}",f'{branches["y"]}=={y_value}&&{mva_window}',"goff")
      # Normalize histogram
      sum_weight = hist.GetSumOfWeights()
      if sum_weight != 0:
        hist.Scale(1/sum_weight)
        display_histograms[branch].append(hist)

  return signal_fraction_bins, display_histograms

def draw_signi_detail(signi_detail_dict):
  # Make significance graphs
  significance_graphs = {}
  significance_min_max = [999,-999]
  integrated_significances = {}
  integrated_significance_errs = {}
  for mva_name in signi_detail_dict:
    integrated_significances[mva_name] = signi_detail_dict[mva_name][0]
    integrated_significance_errs[mva_name] = signi_detail_dict[mva_name][1]
    signal_fraction_bins = signi_detail_dict[mva_name][2]
    significances = signi_detail_dict[mva_name][3]
    significance_errs = signi_detail_dict[mva_name][4]
    mva_err = [0]*len(signal_fraction_bins)
    #significance_graphs[mva_name] = ROOT.TGraph(len(signal_fraction_bins), array.array('d',signal_fraction_bins), array.array('d',significances))
    significance_graphs[mva_name] = ROOT.TGraphErrors(len(signal_fraction_bins), array.array('d',signal_fraction_bins), array.array('d',significances), array.array('d',mva_err), array.array('d',significance_errs))
    # Find min max of graph
    x_min, x_max, y_min, y_max = ctypes.c_double(), ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
    significance_graphs[mva_name].ComputeRange(x_min, y_min, x_max, y_max)
    if y_min.value < significance_min_max[0]: significance_min_max[0] = y_min.value
    if y_max.value > significance_min_max[1]: significance_min_max[1] = y_max.value
  # Draw signifiance graphs
  c1 = measure_tools.new_canvas()
  c1.SetLeftMargin(0.15)
  significance_legend = ROOT.TLegend(0.2, 0.7, 0.7, 0.87)
  colors = [4,800-7,8,1,6,7,8,9,46,49,41,38,30,33]
  for iMva, mva_name in enumerate(signi_detail_dict):
    significance_graphs[mva_name].SetTitle(f';Signal percentile;Bin significance')
    significance_graphs[mva_name].GetYaxis().SetRangeUser(significance_min_max[0]-0.01, significance_min_max[1]+0.01)
    significance_legend.AddEntry(significance_graphs[mva_name], f'{mva_name}, comb. signi= {integrated_significances[mva_name]:.3f}+-{integrated_significance_errs[mva_name]:.3f}')
    if iMva == 0: significance_graphs[mva_name].Draw("APL RX")
    else: significance_graphs[mva_name].Draw("PL RX")
    significance_graphs[mva_name].SetLineColor(colors[iMva])
    significance_graphs[mva_name].SetMarkerColor(colors[iMva])
    significance_graphs[mva_name].SetMarkerStyle(21)
    significance_graphs[mva_name].SetMarkerSize(0.3)
    significance_legend.Draw()
  c1.SaveAs(f'plots/significances_with_resolution.pdf')

  # Make resolution graphs
  signal_width_graphs = {}
  signal_width_min_max = [999,-999]
  for mva_name in signi_detail_dict:
    signal_fraction_bins = signi_detail_dict[mva_name][2]
    signal_widths = [mass_values[3]-mass_values[0] for mass_values in signi_detail_dict[mva_name][5]] # Use 90%
    signal_width_graphs[mva_name] = ROOT.TGraph(len(signal_fraction_bins), array.array('d',signal_fraction_bins), array.array('d',signal_widths))
    # Find min max of graph
    x_min, x_max, y_min, y_max = ctypes.c_double(), ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
    signal_width_graphs[mva_name].ComputeRange(x_min, y_min, x_max, y_max)
    if y_min.value < signal_width_min_max[0]: signal_width_min_max[0] = y_min.value
    if y_max.value > signal_width_min_max[1]: signal_width_min_max[1] = y_max.value
  # Draw signal width graphs
  c3 = measure_tools.new_canvas()
  c3.SetLeftMargin(0.15)
  signal_width_legend = ROOT.TLegend(0.5, 0.7, 0.98, 0.98)
  for iMva, mva_tag in enumerate(signal_width_graphs):
    signal_width_graphs[mva_tag].SetTitle(f';Signal percentile;signal mass window (90%)')
    if iMva == 0: signal_width_graphs[mva_tag].Draw("APL RX")
    else: signal_width_graphs[mva_tag].Draw("PL RX")
    signal_width_graphs[mva_tag].SetLineColor(colors[iMva])
    signal_width_graphs[mva_tag].SetMarkerColor(colors[iMva])
    signal_width_graphs[mva_tag].SetMarkerStyle(21)
    signal_width_graphs[mva_tag].SetMarkerSize(0.3)
    signal_width_graphs[mva_tag].GetYaxis().SetRangeUser(signal_width_min_max[0]-0.1, signal_width_min_max[1]+0.1)
    signal_width_legend.AddEntry(signal_width_graphs[mva_tag], f'{mva_tag}')
    signal_width_legend.Draw()
  c3.SaveAs(f'plots/signal_resolution.pdf')

def draw_signi_detail_train_eval(signi_detail_dict, train_signi_detail_dict):
  # Make significance graphs
  significance_graphs = {}
  train_significance_graphs = {}
  significance_min_max = [999,-999]
  integrated_significances = {}
  integrated_significance_errs = {}
  train_integrated_significances = {}
  train_integrated_significance_errs = {}
  for iMva, mva_name in enumerate(signi_detail_dict):
    # Make graph using eval sample
    integrated_significances[mva_name] = signi_detail_dict[mva_name][0]
    integrated_significance_errs[mva_name] = signi_detail_dict[mva_name][1]
    signal_fraction_bins = signi_detail_dict[mva_name][2]
    significances = signi_detail_dict[mva_name][3]
    significance_errs = signi_detail_dict[mva_name][4]
    mva_err = [0]*len(signal_fraction_bins)
    #significance_graphs[mva_name] = ROOT.TGraph(len(signal_fraction_bins), array.array('d',signal_fraction_bins), array.array('d',significances))
    significance_graphs[mva_name] = ROOT.TGraphErrors(len(signal_fraction_bins), array.array('d',signal_fraction_bins), array.array('d',significances), array.array('d',mva_err), array.array('d',significance_errs))

    # Make graph using train sample
    train_integrated_significances[mva_name] = train_signi_detail_dict[mva_name][0]
    train_integrated_significance_errs[mva_name] = train_signi_detail_dict[mva_name][1]
    train_significances = train_signi_detail_dict[mva_name][3]
    train_significance_errs = signi_detail_dict[mva_name][4]
    train_significance_graphs[mva_name] = ROOT.TGraphErrors(len(signal_fraction_bins), array.array('d',signal_fraction_bins), array.array('d',train_significances), array.array('d',mva_err), array.array('d',train_significance_errs))

    # Find min max of graph
    x_min, x_max, y_min, y_max = ctypes.c_double(), ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
    # For eval sample
    significance_graphs[mva_name].ComputeRange(x_min, y_min, x_max, y_max)
    if y_min.value < significance_min_max[0]: significance_min_max[0] = y_min.value
    if y_max.value > significance_min_max[1]: significance_min_max[1] = y_max.value
    # For train sample
    train_significance_graphs[mva_name].ComputeRange(x_min, y_min, x_max, y_max)
    if y_min.value < significance_min_max[0]: significance_min_max[0] = y_min.value
    if y_max.value > significance_min_max[1]: significance_min_max[1] = y_max.value

    # Draw signifiance graphs
    c1 = measure_tools.new_canvas()
    c1.SetLeftMargin(0.15)
    significance_legend = ROOT.TLegend(0.2, 0.7, 0.7, 0.87)
    colors = [ROOT.kRed,ROOT.kBlack]
    significance_graphs[mva_name].SetTitle(f';Signal percentile;Bin significance')
    significance_graphs[mva_name].GetYaxis().SetRangeUser(significance_min_max[0]-0.01, significance_min_max[1]+0.01)
    significance_legend.AddEntry(significance_graphs[mva_name], f'{mva_name} (eval), comb. signi= {integrated_significances[mva_name]:.3f}+-{integrated_significance_errs[mva_name]:.3f}')
    significance_legend.AddEntry(train_significance_graphs[mva_name], f'{mva_name} (train), comb. signi= {train_integrated_significances[mva_name]:.3f}+-{train_integrated_significance_errs[mva_name]:.3f}')
    significance_graphs[mva_name].Draw("APL RX")
    train_significance_graphs[mva_name].Draw("PL RX")
    significance_graphs[mva_name].SetLineColor(colors[0])
    significance_legend.Draw()
    c1.SaveAs(f'plots/{mva_name}_significances_with_resolution.pdf')

  # Make resolution graphs
  signal_width_graphs = {}
  train_signal_width_graphs = {}
  signal_width_min_max = [999,-999]
  for iMva, mva_name in enumerate(signi_detail_dict):
    # Make graph using eval sample
    signal_fraction_bins = signi_detail_dict[mva_name][2]
    signal_widths = [mass_values[3]-mass_values[0] for mass_values in signi_detail_dict[mva_name][5]] # Use 90%
    signal_width_graphs[mva_name] = ROOT.TGraph(len(signal_fraction_bins), array.array('d',signal_fraction_bins), array.array('d',signal_widths))

    # Make graph using train sample
    train_signal_widths = [mass_values[3]-mass_values[0] for mass_values in train_signi_detail_dict[mva_name][5]] # Use 90%
    train_signal_width_graphs[mva_name] = ROOT.TGraph(len(signal_fraction_bins), array.array('d',signal_fraction_bins), array.array('d',train_signal_widths))

    # Find min max of graph
    x_min, x_max, y_min, y_max = ctypes.c_double(), ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
    signal_width_graphs[mva_name].ComputeRange(x_min, y_min, x_max, y_max)
    if y_min.value < signal_width_min_max[0]: signal_width_min_max[0] = y_min.value
    if y_max.value > signal_width_min_max[1]: signal_width_min_max[1] = y_max.value
    train_signal_width_graphs[mva_name].ComputeRange(x_min, y_min, x_max, y_max)
    if y_min.value < signal_width_min_max[0]: signal_width_min_max[0] = y_min.value
    if y_max.value > signal_width_min_max[1]: signal_width_min_max[1] = y_max.value

    # Draw signal width graphs
    c3 = measure_tools.new_canvas()
    c3.SetLeftMargin(0.15)
    signal_width_legend = ROOT.TLegend(0.5, 0.7, 0.98, 0.98)
    signal_width_graphs[mva_name].SetTitle(f';Signal percentile;signal mass window (90%)')
    signal_width_graphs[mva_name].Draw("APL RX")
    train_signal_width_graphs[mva_name].Draw("PL RX")
    signal_width_graphs[mva_name].SetLineColor(colors[0])
    signal_width_graphs[mva_name].SetMarkerColor(colors[0])
    signal_width_graphs[mva_name].SetMarkerStyle(21)
    signal_width_graphs[mva_name].SetMarkerSize(0.3)
    signal_width_graphs[mva_name].GetYaxis().SetRangeUser(signal_width_min_max[0]-0.1, signal_width_min_max[1]+0.1)
    signal_width_legend.AddEntry(signal_width_graphs[mva_name], f'{mva_name} (eval)')
    signal_width_legend.AddEntry(train_signal_width_graphs[mva_name], f'{mva_name} (train)')
    signal_width_legend.Draw()
    c3.SaveAs(f'plots/{mva_name}_signal_resolution.pdf')

# Used branches: {'y', 'yhat', 'weight', 'observable'}
def calculate_roc(root_filename, tree_name, branches, narrow_mass_window=False, detail_output=False):
  root_file = uproot.open(root_filename)
  y_array = root_file[tree_name][branches['y']].array(library='np')
  yhat_array = root_file[tree_name][branches['yhat']].array(library='np')
  weight_array = root_file[tree_name][branches['weight']].array(library='np')
  observable_array = root_file[tree_name][branches['observable']].array(library='np')
  print('Setting negative weights to zero for roc calculation')
  weight_array[weight_array<0] = 0.
  if narrow_mass_window:
    # Cut on mass
    mass_cut = (observable_array<128).__and__(observable_array>122)
    y_array = y_array[mass_cut]
    yhat_array = yhat_array[mass_cut]
    weight_array = weight_array[mass_cut]
    fpr, tpr, threshold = sklearn.metrics.roc_curve(y_array, yhat_array, sample_weight=weight_array)
  else:
    fpr, tpr, threshold = sklearn.metrics.roc_curve(y_data, yhat_data, sample_weight=weight_array)
  auc = sklearn.metrics.auc(fpr, tpr)*100
  if detail_output: return auc, [fpr, tpr]
  else: return auc

def draw_roc_detail(roc_detail_dict):
  #plt.figure(figsize=(5,5))
  plt.figure(figsize=(7.88,7.49)) # To match with ROOT size
  for mva_name in roc_detail_dict:
    fpr, tpr = roc_detail_dict[mva_name]
    plt.plot(fpr, tpr, lw=1, label=f"{mva_name}, AUC = {sklearn.metrics.auc(fpr, tpr)*100:.1f}%")
  plt.xlabel(r'False positive rate')
  plt.ylabel(r'True positive rate')
  plt.ylim(0,1)
  plt.xlim(0,1)
  plt.legend(loc='lower right')
  filename = "plots/roc.pdf"
  plt.savefig(filename)
  print(f"Saved to {filename}")

def draw_roc_detail_train_eval(roc_detail_dict, train_roc_detail_dict):
  for mva_name in roc_detail_dict:
    #plt.figure(figsize=(5,5))
    plt.figure(figsize=(7.88,7.49)) # To match with ROOT size
    fpr, tpr = roc_detail_dict[mva_name]
    train_fpr, train_tpr = train_roc_detail_dict[mva_name]
    plt.plot(fpr, tpr, lw=1, label=f"{mva_name} (eval), AUC = {sklearn.metrics.auc(fpr, tpr)*100:.1f}%")
    plt.plot(train_fpr, train_tpr, lw=1, label=f"{mva_name} (train), AUC = {sklearn.metrics.auc(train_fpr, train_tpr)*100:.1f}%")
    plt.xlabel(r'False positive rate')
    plt.ylabel(r'True positive rate')
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.legend(loc='lower right')
    filename = f"plots/{mva_name}_roc.pdf"
    plt.savefig(filename)
    print(f"Saved to {filename}")

def draw_mass_shape_difference(mass_difference_detail,name_tag=''):
  for mva_name in mass_difference_detail:
    mass_shape_difference, signal_fraction_bins, mass_histograms = mass_difference_detail[mva_name]
    # Make canvas
    c1 = measure_tools.new_canvas()
    c1.SetLeftMargin(0.15)
    # Find min,max of histograms
    hist_min_max = [999,-999]
    for iBin, hist in enumerate(mass_histograms):
      if hist_min_max[0] > hist.GetMinimum(): hist_min_max[0] = hist.GetMinimum()
      if hist_min_max[1] < hist.GetMaximum(): hist_min_max[1] = hist.GetMaximum()
    # Draw base histogram
    ROOT.gStyle.SetOptStat(0)
    bkg_hist = ROOT.TH1F("bkg_hist",mva_name+";m_{ll#gamma} [GeV];Normalized / 1 GeV",80,100,180)
    bkg_hist.SetMinimum(hist_min_max[0])
    bkg_hist.SetMaximum(hist_min_max[1]*1.1)
    bkg_hist.Draw()
    # Set colors
    colors = [ROOT.kGreen-8, ROOT.kGreen-5, ROOT.kGreen+4,
              ROOT.kBlue-10, ROOT.kBlue-7, 
              ROOT.kRed-10, ROOT.kRed-7, ROOT.kRed,]
    # Draw histogram and legend
    legend = ROOT.TLegend(0.6, 0.5, 0.90, 0.88)
    for iBin, hist in enumerate(mass_histograms):
      hist.SetLineColor(colors[iBin])
      hist.Draw('same')
      legend.AddEntry(hist, f'sig. p = {signal_fraction_bins[iBin]:.2f}', 'l')
    legend.Draw()
    # Draw correlation value
    box = ROOT.TPaveText(0.32, 0.15, 0.67, 0.35, 'NDC NB')
    box.SetFillColorAlpha(0,0)
    box.AddText(f'shape difference: {mass_shape_difference:.4f}')
    box.Draw()
    c1.SaveAs(f"plots/{mva_name}_{name_tag}_hist_mva.pdf")

def draw_feature_shape_difference(feature_difference_detail_dict,name_tag=''):
  for mva_name in feature_difference_detail_dict:
    signal_fraction_bins, feature_histograms = feature_difference_detail_dict[mva_name]
    for branch in feature_histograms:
      branch_histograms = feature_histograms[branch]
      # Make canvas
      c1 = measure_tools.new_canvas()
      c1.SetLeftMargin(0.15)
      # Find min,max of histograms
      hist_min_max = [999,-999]
      for iBin, hist in enumerate(branch_histograms):
        if hist_min_max[0] > hist.GetMinimum(): hist_min_max[0] = hist.GetMinimum()
        if hist_min_max[1] < hist.GetMaximum(): hist_min_max[1] = hist.GetMaximum()
        hist_x_min_max = [hist.GetXaxis().GetXmin(), hist.GetXaxis().GetXmax()]
      # Draw base histogram
      ROOT.gStyle.SetOptStat(0)
      bkg_hist = ROOT.TH1F(f"bkg_hist_{branch}",f"{mva_name};{branch};Normalized",100,hist_x_min_max[0],hist_x_min_max[1])
      bkg_hist.SetMinimum(hist_min_max[0])
      bkg_hist.SetMaximum(hist_min_max[1]*1.1)
      bkg_hist.Draw()
      # Set colors
      if len(branch_histograms) ==4:
        colors = [ROOT.kGreen-8, ROOT.kGreen+4,
                  ROOT.kBlue-7, ROOT.kRed,]
      else:
        colors = [ROOT.kGreen-8, ROOT.kGreen-5, ROOT.kGreen+4,
                  ROOT.kBlue-10, ROOT.kBlue-7, 
                  ROOT.kRed-10, ROOT.kRed-7, ROOT.kRed,]
      ## Draw histogram and legend
      for iBin, hist in enumerate(branch_histograms):
        hist.SetLineColor(colors[iBin])
        hist.Draw('same')
      #legend = ROOT.TLegend(0.6, 0.5, 0.90, 0.88)
      #for iBin, hist in enumerate(branch_histograms):
      #  legend.AddEntry(hist, f'sig. p = {signal_fraction_bins[iBin]:.2f}', 'l')
      #legend.Draw()
      c1.SaveAs(f"plots/{mva_name}_{name_tag}_hist_mva_{measure_tools.slugify(branch)}.pdf")

def draw_overtraining(overtrain_detail_dict):
  for mva_name in overtrain_detail_dict:
    pvalue_signal, pvalue_bkg, nbins, mva_min, mva_max, yhat_signal_train, yhat_bkg_train, yhat_signal_eval, yhat_bkg_eval,residual_y_signal, residual_y_bkg = overtrain_detail_dict[mva_name]
    y_max = np.amax([yhat_signal_eval.GetMaximum(), yhat_signal_train.GetMaximum(), yhat_bkg_eval.GetMaximum(), yhat_bkg_train.GetMaximum()])
    blank_hist = ROOT.TH1F('blank_hist','', nbins, mva_min, mva_max)
    blank_hist.SetMaximum(y_max * 1.1)
    blank_hist.SetMinimum(0.)
    c1 = measure_tools.new_canvas()
    up_pad = ROOT.TPad('up_pad', 'up_pad', 0, 0.3, 1, 1)
    low_pad = ROOT.TPad('low_pad', 'low_pad', 0, 0, 1, 0.3)
    up_pad.Draw()
    low_pad.Draw()
    up_pad.cd()
    up_pad.SetTopMargin(0.1)
    up_pad.SetRightMargin(0.08)
    up_pad.SetLeftMargin(0.24)
    up_pad.SetBottomMargin(0.22)
    ROOT.gStyle.SetOptStat(0)
    blank_hist.GetXaxis().SetTitle("Network response")
    blank_hist.GetYaxis().SetTitle("Normalized")
    blank_hist.GetXaxis().SetTitleSize(0.09)
    blank_hist.GetXaxis().SetLabelSize(0.09)
    blank_hist.GetYaxis().SetTitleSize(0.09)
    blank_hist.GetYaxis().SetLabelSize(0.09)
    blank_hist.GetXaxis().SetLabelOffset(0.02)
    blank_hist.GetXaxis().SetTitleOffset(1.2)
    blank_hist.SetMinimum(0.)
    blank_hist.GetXaxis().SetNdivisions(5,4,0)
    blank_hist.GetYaxis().SetNdivisions(4,4,0)
    blank_hist.Draw()
    yhat_bkg_eval.SetLineColor(ROOT.kBlue)
    yhat_bkg_eval.SetFillColor(ROOT.kBlue)
    yhat_bkg_eval.SetFillStyle(3954)
    yhat_bkg_eval.Draw('same hist')
    yhat_bkg_train.SetMarkerStyle(5)
    yhat_bkg_train.SetMarkerSize(1.)
    yhat_bkg_train.SetMarkerColor(ROOT.kBlue)
    yhat_bkg_train.Draw('same')
    yhat_signal_eval.SetLineColor(ROOT.kRed)
    yhat_signal_eval.SetFillColor(ROOT.kRed)
    yhat_signal_eval.SetFillStyle(3945)
    yhat_signal_eval.Draw('hist same')
    yhat_signal_train.SetMarkerStyle(5)
    yhat_signal_train.SetMarkerSize(1.)
    yhat_signal_train.SetMarkerColor(ROOT.kRed)
    yhat_signal_train.Draw('same P')
    legend = ROOT.TLegend(0.7, 0.9, 0.9, 0.98)
    legend.AddEntry(yhat_signal_eval,  "Signal (eval  sample)", "f")
    legend.AddEntry(yhat_signal_train, "Signal (train sample)", 'p')
    legend.AddEntry(yhat_bkg_eval,     "Background (eval  sample)", 'f')
    legend.AddEntry(yhat_bkg_train,    "Background (train sample)", 'p')
    legend.Draw()
    low_pad.cd()
    low_pad.SetTopMargin(0.1)
    low_pad.SetRightMargin(0.08)
    low_pad.SetLeftMargin(0.24)
    low_pad.SetBottomMargin(0.22)
    n_bins = blank_hist.GetNbinsX()
    residual_x = array.array('d',[0.]*n_bins)
    for index in range(n_bins):
      residual_x[index] = mva_min + (mva_max-mva_min)/n_bins * index + (mva_max-mva_min)/n_bins/2
    ROOT.gErrorIgnoreLevel = ROOT.kError
    ROOT.gErrorIgnoreLevel = ROOT.kPrint
    residual_signal = ROOT.TGraph(n_bins, residual_x, residual_y_signal)
    # Set range on computed graph. Set twice because TGraph's axis looks strange otherwise
    residual_signal.GetXaxis().SetLimits(mva_min, mva_max)
    residual_signal.SetTitle('')
    residual_signal.GetYaxis().SetRangeUser(-3.5, 3.5)
    residual_signal.GetYaxis().SetTitle("Residuals")
    residual_signal.GetYaxis().CenterTitle()
    residual_signal.GetYaxis().SetTitleOffset(0.60)
    residual_signal.GetYaxis().SetTitleSize(0.20)
    residual_signal.GetYaxis().SetLabelSize(0.20)
    residual_signal.GetXaxis().SetLabelSize(0.00)
    residual_signal.GetYaxis().SetNdivisions(2,0,0,True)
    residual_signal.SetMarkerStyle(21)
    residual_signal.SetMarkerSize(0.3)
    residual_signal.SetMarkerColor(ROOT.kRed)
    residual_signal.Draw("AP")
    residual_bkg = ROOT.TGraph(n_bins, residual_x, residual_y_bkg)
    # Set range on computed graph. Set twice because TGraph's axis looks strange otherwise
    residual_bkg.GetXaxis().SetLimits(mva_min, mva_max)
    residual_bkg.SetTitle('')
    residual_bkg.GetYaxis().SetRangeUser(-3.5, 3.5)
    residual_bkg.GetYaxis().SetTitle("Normalized residuals")
    residual_bkg.GetYaxis().CenterTitle()
    residual_bkg.SetMarkerStyle(21)
    residual_bkg.SetMarkerSize(0.3)
    residual_bkg.SetMarkerColor(ROOT.kBlue)
    residual_bkg.Draw("P same")
    zero_line = ROOT.TLine(mva_min, 0, mva_max, 0)
    zero_line.Draw()
    box = ROOT.TPaveText(0.25, 0.75, 0.90, 0.88, 'NDC NB')
    box.SetFillColorAlpha(0,0)
    box.AddText('#chi^{2} eval p-value, signal='+f'{pvalue_signal:.3f} background={pvalue_bkg:.3f}')
    box.Draw()
    c1.SaveAs(f'plots/{mva_name}_sig_bkg_overtrain.pdf')

# Used branches: {'y', 'yhat', 'weight', 'observable', 'sample_id', 'bkg_ids', 'sig_ids'}
def calculate_binned_significance(root_filename, tree_name, branches, mva_nbins=5, throw_away_bin=0, detail_output=False):
  base_cut = '1'
  #base_cut = 'x[11]>0.8'
  #mass_window = [122, 128]
  mass_window = [120, 130]
  chain = ROOT.TChain(tree_name)
  chain.Add(root_filename)
  mva_min = chain.GetMinimum(branches['yhat'])
  mva_max = chain.GetMaximum(branches['yhat'])
  # Make mva historams
  mva_hists = {}
  for bkg_id in branches['bkg_ids']: mva_hists[bkg_id] = ROOT.TH1D(f'mva_hist_id{bkg_id}', 'bkg_hist_id{bkg_id}', 100, mva_min, mva_max)
  for sig_id in branches['sig_ids']: mva_hists[sig_id] = ROOT.TH1D(f'mva_hist_id{sig_id}', 'sig_hist_id{sig_id}', 100, mva_min, mva_max)
  bkg_hist = ROOT.TH1D('bkg_hist', 'bkg_hist', 100, mva_min, mva_max)
  sig_hist = ROOT.TH1D('sig_hist', 'sig_hist', 100, mva_min, mva_max)
  # Draw mva histograms
  for sample_id in mva_hists:
    chain.Draw(f"{branches['yhat']}>>mva_hist_id{sample_id}",f"({branches['observable']}>{mass_window[0]}&&{branches['observable']}<{mass_window[1]}&&{branches['sample_id']}=={sample_id}&&{base_cut})*{branches['weight']}","goff")
  chain.Draw(f"{branches['yhat']}>>bkg_hist",f"({branches['observable']}>{mass_window[0]}&&{branches['observable']}<{mass_window[1]}&&{branches['y']}==0&&{base_cut})*{branches['weight']}","goff")
  chain.Draw(f"{branches['yhat']}>>sig_hist",f"({branches['observable']}>{mass_window[0]}&&{branches['observable']}<{mass_window[1]}&&{branches['y']}==1&&{base_cut})*{branches['weight']}","goff")
  # Find bin edges
  ROOT.gInterpreter.ProcessLine('.L ../root_scripts/evaluate_mva.C+')
  # results = [(mva_nbins, bin_edges, significance, significance_err, significances, significance_errs, signal_yield, background_yield)]
  if detail_output == False: max_bins = mva_nbins
  else: max_bins = mva_nbins+1
  # Set min signal to 10% of total signal yield
  signal_10percent = sig_hist.GetSum()/10
  bkg_hist.Smooth(10)
  sig_hist.Smooth(10)
  print(f'Optimizing bins with min signal yeild of {signal_10percent:.2f}, which is 10% of signal.')  
  results = ROOT.binning_optimizer(sig_hist, bkg_hist, max_bins, signal_10percent, 1., throw_away_bin) # mva_nbins, min_signal_yield, scale, cut away bins
  # Return results
  if detail_output == False: return results[mva_nbins-1][2], results[mva_nbins-1][3] # significance, significance_err
  else: 
    # Draw lly_m for each bin
    obs_min = chain.GetMinimum(branches['observable'])
    obs_max = chain.GetMaximum(branches['observable'])
    # obs_hists[sample_id] = [obs_hist for bin, ...]
    obs_hists = {}
    for sample_id in mva_hists: obs_hists[sample_id] = []
    sig_obs_hists = []
    bkg_obs_hists = []
    bin_edges = results[mva_nbins-1][1]
    for ibin in range(mva_nbins):
      if ibin == 0: mva_cut = f'{branches["yhat"]}<{bin_edges[0]}'
      elif ibin == mva_nbins-1: mva_cut = f'{branches["yhat"]}>{bin_edges[mva_nbins-2]}'
      else: mva_cut = f'{branches["yhat"]}>{bin_edges[ibin-1]}&&{branches["yhat"]}<{bin_edges[ibin]}'
      # Draw obs_hist
      for sample_id in mva_hists: 
        obs_hist = ROOT.TH1D(f'obs_hist_id{sample_id}_bin{ibin}',f'obs_hist_id{sample_id}_bin{ibin}',100,obs_min,obs_max)
        chain.Draw(f"{branches['observable']}>>obs_hist_id{sample_id}_bin{ibin}",f"({mva_cut}&&{branches['sample_id']}=={sample_id}&&{base_cut})*{branches['weight']}","goff")
        obs_hists[sample_id].append(obs_hist)
      sig_obs_hist = ROOT.TH1D(f'sig_hist_bin{ibin}',f'sig_hist_bin{ibin}',100,obs_min,obs_max)
      chain.Draw(f"{branches['observable']}>>sig_hist_bin{ibin}",f"({mva_cut}&&{branches['y']}==1&&{base_cut})*{branches['weight']}","goff")
      sig_obs_hists.append(sig_obs_hist)
      bkg_obs_hist = ROOT.TH1D(f'bkg_hist_bin{ibin}',f'bkg_hist_bin{ibin}',100,obs_min,obs_max)
      chain.Draw(f"{branches['observable']}>>bkg_hist_bin{ibin}",f"({mva_cut}&&{branches['y']}==0&&{base_cut})*{branches['weight']}","goff")
      bkg_obs_hists.append(bkg_obs_hist)
    return results[mva_nbins-1][2], results[mva_nbins-1][3], [mva_nbins, results, sig_hist, bkg_hist, sig_obs_hists, bkg_obs_hists, mva_hists, obs_hists, branches['sig_ids'], branches['bkg_ids']]

# target_bins, results, sig_hist, bkg_hist, sig_obs_hists, bkg_obs_hists, mva_hists, obs_hists
def draw_binned_signi_detail(binned_signi_detail_dict, name_tag=''):
  if name_tag != '': _name_tag = '_'+name_tag
  else: _name_tag = ''
  for mva_name in binned_signi_detail_dict:
    # Draw (signal hist, bkg hist, location of cuts, significances, signal yield, bkg yield, total significance)
    mva_nbins, binned_signi_detail, sig_hist, bkg_hist, sig_obs_hists, bkg_obs_hists, mva_hists, obs_hists, sig_ids, bkg_ids = binned_signi_detail_dict[mva_name]
    nbins, bin_edges, significance, significance_err, significances, significance_errs, signal_yield, background_yield = binned_signi_detail[mva_nbins-1]
    #print(nbins, bin_edges, significance, significances, signal_yield, background_yield)

    # Draw mva signal and background
    c1 = measure_tools.new_canvas()
    c1.SetLeftMargin(0.15)
    sig_hist.SetLineColor(ROOT.kRed)
    sig_hist.SetTitle(';MVA value;Normalized')
    measure_tools.normalize_hist(sig_hist)
    measure_tools.normalize_hist(bkg_hist)
    ROOT.gStyle.SetOptStat(0)
    sig_hist.Draw()
    bkg_hist.Draw('same')
    # Draw bin cuts
    y_max = sig_hist.GetMaximum()
    edge_lines = []
    for iEdge in range(len(bin_edges)):
      edge_line = ROOT.TLine(bin_edges[iEdge],0,bin_edges[iEdge],y_max)
      edge_lines.append(edge_line)
      edge_line.Draw()
    # Significance box
    box = ROOT.TPaveText(0.2, 0.8, 0.5, 1., 'NDC NB')
    box.AddText(f'{mva_nbins}-bin signi.: {significance:.3f}+-{significance_err:.3f}')
    for ibin in range(mva_nbins):
      box.AddText(f'bin{ibin+1} signi:{significances[ibin]:.3f}+-{significance_errs[ibin]:.3f}, sig:{signal_yield[ibin]:.1f}, bkg:{background_yield[ibin]:.1f}')
    box.Draw()
    # Legend box
    legend = ROOT.TLegend(0.6, 0.85, 0.95, 0.95)
    legend.AddEntry(sig_hist, 'signal')
    legend.AddEntry(bkg_hist, 'background')
    legend.Draw()
    c1.SaveAs(f'plots/{mva_name}_binned_mva{_name_tag}.pdf')

    # Draw mva sample
    c3 = measure_tools.new_canvas()
    c3.SetLeftMargin(0.15)
    colors = [4,800-7,8,1,6,7,8,9,46,49,41,38,30,33]
    for isample, sample_id in enumerate(mva_hists):
      mva_hist = mva_hists[sample_id]
      mva_hist.SetLineColor(colors[isample])
      mva_hist.SetTitle(';MVA value;Normalized')
      measure_tools.normalize_hist(mva_hist)
      if isample==0: mva_hist.Draw()
      else: mva_hist.Draw('same')
    set_max_th1()
    # Draw bin cuts
    edge_lines = []
    for iEdge in range(len(bin_edges)):
      edge_line = ROOT.TLine(bin_edges[iEdge],0,bin_edges[iEdge],y_max)
      edge_lines.append(edge_line)
      edge_line.Draw()
    # Significance box
    box = ROOT.TPaveText(0.2, 0.8, 0.5, 1., 'NDC NB')
    if name_tag != '':box.AddText(f'({name_tag}) {mva_nbins}-bin signi.: {significance:.3f}+-{significance_err:.3f}')
    else: box.AddText(f'{mva_nbins}-bin signi.: {significance:.3f}+-{significance_err:.3f}')
    for ibin in range(mva_nbins):
      box.AddText(f'bin{ibin+1} signi:{significances[ibin]:.3f}+-{significance_errs[ibin]:.3f}, sig:{signal_yield[ibin]:.1f}, bkg:{background_yield[ibin]:.1f}')
    box.Draw()
    # Legend box
    legend = ROOT.TLegend(0.6, 0.90, 0.95, 0.98)
    for sample_id in mva_hists:
      legend.AddEntry(mva_hists[sample_id], f'sample{sample_id}')
    legend.Draw()
    c3.SaveAs(f'plots/{mva_name}_binned_mva_per_sample{_name_tag}.pdf')
    
    # Draw (lly_m hist for each bin)
    scale_signal = 10
    for ibin in range(mva_nbins):
      c2 = measure_tools.new_canvas()
      # Draw hists
      bkg_obs_hists[ibin].SetTitle(';m_{ll#gamma} [GeV]; Entries')
      bkg_obs_hists[ibin].Draw()
      sig_obs_hists[ibin].SetLineColor(ROOT.kRed)
      sig_obs_hists[ibin].Scale(scale_signal)
      sig_obs_hists[ibin].Draw('same')
      # Draw legend
      legend = ROOT.TLegend(0.6, 0.85, 0.95, 0.95)
      legend.AddEntry(sig_obs_hists[ibin], f'signalx{scale_signal}')
      legend.AddEntry(bkg_obs_hists[ibin], 'background')
      legend.Draw()
      # Draw stats
      box = ROOT.TPaveText(0.2, 0.9, 0.5, 0.98, 'NDC NB')
      box.SetFillColorAlpha(0,0)
      box.AddText(f'Total signi.: {significance:.3f}+-{significance_err:.3f}')
      box.AddText(f'bin{ibin+1} signi.: {significances[ibin]:.3f}+-{significance_errs[ibin]:.3f}')
      box.Draw()
      # Draw legend
      c2.SaveAs(f'plots/{mva_name}_binned_mva_mlly_bin{ibin}{_name_tag}.pdf')

    # Draw (lly_m hist for each bin) for each sample
    scale_signal = 10
    for ibin in range(mva_nbins):
      c4 = measure_tools.new_canvas()
      c4.SetLeftMargin(0.15)
      stack_bkg = ROOT.THStack(f'obs_hist_bkg_stack_bin[ibin]','obs_hist_bkg_stack_bin[ibin]')
      # Add bkg to stack
      for isample, sample_id in enumerate(obs_hists):
        if sample_id in bkg_ids:
          obs_hist = obs_hists[sample_id][ibin]
          obs_hist.SetLineColor(colors[isample])
          obs_hist.Rebin(2)
          stack_bkg.Add(obs_hist)
      # Draw bkg stack
      stack_bkg.SetTitle(';m_{ll#gamma} [GeV]; Entries')
      stack_bkg.Draw()
      # Draw signal stack
      for isample, sample_id in enumerate(obs_hists):
        if sample_id in bkg_ids: continue
        obs_hist = obs_hists[sample_id][ibin]
        obs_hist.SetLineColor(colors[isample])
        obs_hist.Scale(scale_signal)
        obs_hist.Rebin(2)
        obs_hist.Draw("same")
      # Draw legend
      legend = ROOT.TLegend(0.6, 0.85, 0.95, 0.95)
      for isample, sample_id in enumerate(obs_hists):
        obs_hist = obs_hists[sample_id][ibin]
        if sample_id in bkg_ids: legend.AddEntry(obs_hist, f'sample{sample_id}')
        else: legend.AddEntry(obs_hist, f'sample{sample_id}x{scale_signal}')
      legend.Draw()
      # Draw stats
      box = ROOT.TPaveText(0.2, 0.9, 0.5, 0.98, 'NDC NB')
      box.SetFillColorAlpha(0,0)
      box.AddText(f'Total signi.: {significance:.3f}+-{significance_err:.3f}')
      box.AddText(f'bin{ibin+1} signi.: {significances[ibin]:.3f}+-{significance_errs[ibin]:.3f}')
      box.Draw()
      # Draw legend
      c4.SaveAs(f'plots/{mva_name}_binned_mva_mlly_bin{ibin}_per_sample{_name_tag}.pdf')

# target_bins, results, sig_hist, bkg_hist, sig_obs_hists, bkg_obs_hists, mva_hists, obs_hists
def draw_binned_signi_detail_train_eval(binned_signi_detail_dict, train_binned_signi_detail_dict):
  for mva_name in binned_signi_detail_dict:
    # Draw (signal hist, bkg hist, location of cuts, significances, signal yield, bkg yield, total significance)
    mva_nbins, binned_signi_detail, sig_hist, bkg_hist, sig_obs_hists, bkg_obs_hists, mva_hists, obs_hists, sig_ids, bkg_ids = binned_signi_detail_dict[mva_name]
    nbins, bin_edges, significance, significance_err, significances, significance_errs, signal_yield, background_yield = binned_signi_detail[mva_nbins-1]
    #print(nbins, bin_edges, significance, significances, signal_yield, background_yield)
    train_mva_nbins, train_binned_signi_detail, train_sig_hist, train_bkg_hist, train_sig_obs_hists, train_bkg_obs_hists, train_mva_hists, train_obs_hists, train_sig_ids, train_bkg_ids = train_binned_signi_detail_dict[mva_name]
    train_nbins, train_bin_edges, train_significance, train_significance_err, train_significances, train_significance_errs, train_signal_yield, train_background_yield = train_binned_signi_detail[train_mva_nbins-1]

    # Get significance graph
    bin_errs = [0]*len(significances)
    binned_signi_graph = ROOT.TGraphErrors(len(significances), array.array('d',[ibin for ibin in range(len(significances))]), array.array('d',significances), array.array('d',bin_errs), array.array('d',significance_errs))
    train_binned_signi_graph = ROOT.TGraphErrors(len(train_significances), array.array('d',[ibin for ibin in range(len(train_significances))]), array.array('d',train_significances), array.array('d',bin_errs),array.array('d',train_significance_errs))

    # Find min max of graph
    x_min, x_max, y_min, y_max = ctypes.c_double(), ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
    # For eval sample
    significance_min_max = [999,-999]
    binned_signi_graph.ComputeRange(x_min, y_min, x_max, y_max)
    if y_min.value < significance_min_max[0]: significance_min_max[0] = y_min.value
    if y_max.value > significance_min_max[1]: significance_min_max[1] = y_max.value
    # For train sample
    train_binned_signi_graph.ComputeRange(x_min, y_min, x_max, y_max)
    if y_min.value < significance_min_max[0]: significance_min_max[0] = y_min.value
    if y_max.value > significance_min_max[1]: significance_min_max[1] = y_max.value

    # Plot significance graph
    c1 = measure_tools.new_canvas()
    c1.SetLeftMargin(0.15)
    significance_legend = ROOT.TLegend(0.2, 0.9, 0.99, 0.99)
    colors = [ROOT.kRed,ROOT.kBlack]
    binned_signi_graph.SetTitle(f';ibin;Bin significance')
    binned_signi_graph.GetXaxis().SetLimits(x_min.value-0.5, x_max.value+0.5)
    binned_signi_graph.GetYaxis().SetRangeUser(significance_min_max[0]-0.01, significance_min_max[1]+0.01)
    bin_edges_string = ','.join([f'{bin_edge:.2f}' for bin_edge in bin_edges])
    significance_legend.AddEntry(binned_signi_graph, f'{mva_name} (eval), comb. signi= {significance:.3f}+-{significance_err:.3f}, edges={bin_edges_string}')
    train_bin_edges_string = ','.join([f'{bin_edge:.2f}' for bin_edge in train_bin_edges])
    significance_legend.AddEntry(train_binned_signi_graph, f'{mva_name} (train), comb. signi= {train_significance:.3f}+-{train_significance_err:.3f}, edges={train_bin_edges_string}')
    binned_signi_graph.Draw("APL")
    train_binned_signi_graph.Draw("PL")
    binned_signi_graph.SetLineColor(colors[0])
    binned_signi_graph.SetMarkerColor(colors[0])
    significance_legend.Draw()
    c1.SaveAs(f'plots/{mva_name}_binned_significances.pdf')

# auc_sci_detail = (auc, sci, bins, areas, signis)
def draw_auc_sci_detail(auc_sci_detail_dict,name_tag=''):
  if name_tag != '': name_tag = '_'+name_tag
  for mva_name in auc_sci_detail_dict:
    auc, sci, bins, areas, signis = auc_sci_detail_dict[mva_name]
    integrated_areas = []
    for iarea, area in enumerate(areas):
      if iarea==0: integrated_areas.append(area)
      else: integrated_areas.append(integrated_areas[iarea-1]+area)
    c1 = measure_tools.new_canvas()
    c1.SetLeftMargin(0.15)
    auc_graph = ROOT.TGraph(len(bins), array.array('d',bins), array.array('d',areas))
    auc_graph.SetTitle("AUC bins;mva value;int_{i}^{end}(signal) #times bkg[i]")
    auc_graph.Draw()
    c1.SaveAs(f"plots/{mva_name}_auc_graph{name_tag}.pdf")
    c2 = measure_tools.new_canvas()
    c2.SetLeftMargin(0.15)
    sci_graph = ROOT.TGraph(len(bins), array.array('d',bins), array.array('d',signis))
    sci_graph.SetTitle("SCI bins;mva value;signal/sqrt(bkg)")
    sci_graph.Draw()
    c2.SaveAs(f"plots/{mva_name}_signis_graph{name_tag}.pdf")
    c3 = measure_tools.new_canvas()
    c3.SetLeftMargin(0.15)
    int_auc_graph = ROOT.TGraph(len(bins), array.array('d',bins), array.array('d',integrated_areas))
    int_auc_graph.SetTitle("AUC bins;mva value;int_{i}^{end}(signal) #times bkg[i]")
    int_auc_graph.Draw()
    c3.SaveAs(f"plots/{mva_name}_int_auc_graph{name_tag}.pdf")

# auc_sci_detail = (auc, sci, bins, areas, signis)
def draw_sci_detail_train_eval(auc_sci_detail_dict, train_auc_sci_detail):
  for mva_name in auc_sci_detail_dict:
    auc, sci, bins, areas, signis = auc_sci_detail_dict[mva_name]
    train_auc, train_sci, train_bins, train_areas, train_signis = train_auc_sci_detail_dict[mva_name]
    c2 = measure_tools.new_canvas()
    c2.SetLeftMargin(0.15)
    sci_graph = ROOT.TGraph(len(bins), array.array('d',bins), array.array('d',signis))
    sci_graph.SetLineColor(ROOT.kRed)
    train_sci_graph = ROOT.TGraph(len(bins), array.array('d',bins), array.array('d',train_signis))
    train_sci_graph.SetTitle("SCI bins;mva value;signal/sqrt(bkg)")
    train_sci_graph.Draw()
    sci_graph.Draw("same")
    legend = ROOT.TLegend(0.2, 0.85, 0.5, 0.95)
    legend.AddEntry(sci_graph, f'Eval')
    legend.AddEntry(train_sci_graph, f'Train')
    legend.Draw()
    c2.SaveAs(f"plots/{mva_name}_signis_graph_train_eval.pdf")

# Used branches: {'yhat', 'observable'}
def calculate_obs_mva_correlation(root_filename, tree_name, branches):
  root_filename = mva_filename
  tree_name = 'eval_tree_baseline'
  root_file = uproot.open(root_filename)
  yhat_array = root_file[tree_name][branches['yhat']].array(library='np')
  observable_array = root_file[tree_name][branches['observable']].array(library='np')
  return np.corrcoef(yhat_array, observable_array)[0][1]

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
    'standard_tmva_bdt':['mva_output/standard_tmva_bdt_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'y_ptmass']}],
    #'standard_tmva_bdt_hig19014':['mva_output/standard_tmva_bdt_hig19014_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta']}],
    #'tmva_bdt_decorr':['mva_output/tmva_bdt_decorr_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva', 'yl_drmin', 'yl_drmax', 'cosThetamass3', 'costheta', 'phi', 'lly_ptmass', 'y_eta', 'l1_eta', 'l2_eta', 'l1_ptmass', 'l2_pt']}],
    #'tmva_bdt_windowcut_hig19014':['mva_output/tmva_bdt_windowcut_hig19014_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'y_ptmass']}],
    #'tmva_bdt_windowcut_decorr_hig19014':['mva_output/tmva_bdt_windowcut_decorr_hig19014_results.root', {'y':'classID', 'yhat':'BDT', 'weight': 'weightXyear', 'observable':'lly_m', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva', 'yl_drmin', 'yl_drmax', 'cosThetamass3', 'costheta', 'phi', 'lly_ptmass', 'y_eta', 'l1_eta', 'l2_eta', 'l1_ptmass', 'l2_pt']}],
    #'standard_xgboost':['mva_output/standard_xgboost_results.root', {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'y_ptmass']}],
    #'standard_xgboost_hig19014':['mva_output/standard_xgboost_hig19014_results.root', {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta']}],
    #'xgboost_decorr':['mva_output/xgboost_decorr_results.root', {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva', 'yl_drmin', 'yl_drmax', 'cosThetamass3', 'costheta', 'phi', 'lly_ptmass', 'y_eta', 'l1_eta', 'l2_eta', 'l1_ptmass', 'l2_pt']}],
    #'xgboost_windowcut_hig19014':['mva_output/xgboost_windowcut_hig19014_results.root', {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'y_ptmass']}],
    #'xgboost_windowcut_decorr_hig19014':['mva_output/xgboost_windowcut_decorr_hig19014_results.root', {'y':'y', 'yhat':'yhat', 'weight': 'weight', 'observable':'mass', 'sample_id':'sampleID', 'bkg_ids':[1,2], 'sig_ids':[3], 'x':['y_mva', 'yl_drmin', 'yl_drmax', 'cosThetamass3', 'costheta', 'phi', 'lly_ptmass', 'y_eta', 'l1_eta', 'l2_eta', 'l1_ptmass', 'l2_pt']}],
  }


  # Measure performance
  for mva_name in mva_info:
    mva_filename = mva_info[mva_name][0]
    branches = mva_info[mva_name][1]

    integrated_significance, integrated_significance_err, signi_detail = measure_tools.calculate_significance(mva_filename, 'eval_tree_baseline', branches, fixed_width=False, detail_output=True)
    print(f'[{mva_name}] Total significance: {integrated_significance:.3f}+-{integrated_significance_err:.3f}')
    train_integrated_significance, train_integrated_significance_err, train_signi_detail = measure_tools.calculate_significance(mva_filename, 'train_tree_baseline', branches, fixed_width=False, detail_output=True)
    print(f'[{mva_name}] Total significance (train): {train_integrated_significance:.3f}+-{train_integrated_significance_err:.3f}')
    auc, roc_detail = calculate_roc(mva_filename, 'eval_tree', branches, narrow_mass_window=True, detail_output=True)
    print(f'[{mva_name}] AUC: {auc:.1f}%')
    train_auc, train_roc_detail = calculate_roc(mva_filename, 'train_tree', branches, narrow_mass_window=True, detail_output=True)
    print(f'[{mva_name}] AUC(train): {train_auc:.1f}%')
    bkg_mass_shape_difference, bkg_mass_difference_detail = measure_tools.calculate_mass_shape_difference(mva_filename, 'eval_tree_baseline', branches, detail_output=True)
    print(f'[{mva_name}] Mass shape difference: {bkg_mass_shape_difference:.4f}')
    obs_mva_correlation = calculate_obs_mva_correlation(mva_filename, 'eval_tree_baseline', branches)
    print(f'[{mva_name}] Observable-MVA correlation: {obs_mva_correlation:.3f}')
    pvalue_signal, pvalue_bkg, overtrain_detail = measure_tools.calculate_overtraining(mva_filename, 'train_tree', 'eval_tree', branches, narrow_mass_window=True, detail_output=True)
    print(f'[{mva_name}] chi-2 pvalue (test,eval): signal: {pvalue_signal:.4f} background: {pvalue_bkg:.4f}')
    binned_significance, binned_significance_err, binned_signi_detail = calculate_binned_significance(mva_filename, 'eval_tree_baseline', branches, detail_output=True, mva_nbins=4, throw_away_bin=0)
    print(f'[{mva_name}] Significance 4 bins: {binned_significance:.3f} +- {binned_significance_err:.3f}')
    train_binned_significance, train_binned_significance_err, train_binned_signi_detail = calculate_binned_significance(mva_filename, 'train_tree_baseline', branches, detail_output=True, mva_nbins=4, throw_away_bin=0)
    print(f'[{mva_name}] Significance 4 bins (train): {train_binned_significance:.3f} +- {train_binned_significance_err:.3f}')
    auc_sci, auc_sci_detail = measure_tools.calculate_auc_sci(mva_filename, 'eval_tree', branches, detail_output=True)
    print(f'[{mva_name}] AUC: {auc_sci[0]*100:.1f}% CSI: {auc_sci[1]:.2f}')
    train_auc_sci, train_auc_sci_detail = measure_tools.calculate_auc_sci(mva_filename, 'train_tree', branches, detail_output=True)
    print(f'[{mva_name}] AUC (train): {auc_sci[0]*100:.1f}% CSI (train): {auc_sci[1]:.2f}')
    # Calculation for plotting
    sig_mass_shape_difference, sig_mass_difference_detail = measure_tools.calculate_mass_shape_difference(mva_filename, 'eval_tree_baseline', branches, y_value=1,detail_output=True)
    bkg_feature_shape_difference = find_feature_shape_difference(mva_filename, 'eval_tree_baseline', branches, y_value=0)
    sig_feature_shape_difference = find_feature_shape_difference(mva_filename, 'eval_tree_baseline', branches, y_value=1)
    bkg_feature_shape_difference_in_bin = find_feature_shape_difference_in_bins(mva_filename, 'eval_tree_baseline', branches, mva_bins=binned_signi_detail[1][3][1], y_value=0)
    sig_feature_shape_difference_in_bin = find_feature_shape_difference_in_bins(mva_filename, 'eval_tree_baseline', branches, mva_bins=train_binned_signi_detail[1][3][1], y_value=1)

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
  draw_signi_detail(signi_detail_dict)
  draw_signi_detail_train_eval(signi_detail_dict, train_signi_detail_dict)
  draw_roc_detail(roc_detail_dict)
  draw_roc_detail_train_eval(roc_detail_dict, train_roc_detail_dict)
  draw_mass_shape_difference(bkg_mass_difference_detail_dict,name_tag='bkg')
  draw_mass_shape_difference(sig_mass_difference_detail_dict,name_tag='sig')
  draw_overtraining(overtrain_detail_dict)
  draw_binned_signi_detail(binned_signi_detail_dict)
  draw_binned_signi_detail(train_binned_signi_detail_dict,name_tag='train')
  draw_binned_signi_detail_train_eval(binned_signi_detail_dict, train_binned_signi_detail_dict)
  draw_auc_sci_detail(auc_sci_detail_dict)
  draw_sci_detail_train_eval(auc_sci_detail_dict, train_auc_sci_detail)
  draw_feature_shape_difference(bkg_feature_difference_detail_dict,name_tag='bkg_8bin')
  draw_feature_shape_difference(sig_feature_difference_detail_dict,name_tag='sig_8bin')
  draw_feature_shape_difference(bkg_feature_difference_in_bin_detail_dict,name_tag='bkg_4bin')
  draw_feature_shape_difference(sig_feature_difference_in_bin_detail_dict,name_tag='sig_4bin')

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
