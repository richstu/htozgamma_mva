#!/usr/bin/env python3
import ROOT
ROOT.gROOT.SetBatch(ROOT.kTRUE)
import math
import numpy as np
#from RootDataset import RootDataset
import sklearn.metrics
import uproot
import matplotlib.pyplot as plt
import array
#import slugify
import ctypes
import re
import os

def get_unique_name(name, itrial = 0):
  if ROOT.gROOT.FindObject(name):
    name = f'{name}_{itrial}'
    get_unique_name(name, itrial+1)
  else: return name
def new_canvas(name = "", size = 500):
  canvases = ROOT.gROOT.GetListOfCanvases()
  iCanvas = canvases.GetEntries()
  if name == "":
    canvas_name = f"c_g_{iCanvas}"
  else: canvas_name = name
  canvas_name = get_unique_name(name)
  return ROOT.TCanvas(canvas_name, canvas_name, size, size)

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

import unicodedata
import re
def slugify(string):
  return re.sub(r'[-\s]+', '-',re.sub(r'[^\w\s-]', '',
                  unicodedata.normalize('NFKD', string)).strip().lower())

def normalize_hist(hist):
  sum_weight = hist.GetSumOfWeights()
  hist.Scale(1/sum_weight)

def find_signal_fraction_thresholds(signal_fractions, tmva_chain, mass_name, mva_name, label_name, weight_name = ''):
  # Find min max of variable
  min_val = tmva_chain.GetMinimum(mva_name)
  max_val = tmva_chain.GetMaximum(mva_name)
  #print(f'find signal fraction min: {min_val}, max: {max_val}')
  hist_mva = ROOT.TH1F("hist_mva","hist_mva",10000,min_val*0.9,max_val*1.1)
  if weight_name == '': n_entries = tmva_chain.Draw(mva_name+">>hist_mva", label_name+"==1", 'goff')
  else: n_entries = tmva_chain.Draw(mva_name+">>hist_mva", f'({label_name}==1)*{weight_name}', 'goff')
  mva_quantiles = array.array('d', [0.]*len(signal_fractions))
  mva_fractions = array.array('d', [1.-signal_fraction for signal_fraction in signal_fractions])
  hist_mva.GetQuantiles(len(signal_fractions), mva_quantiles, mva_fractions)
  #print(mva_fractions,mva_quantiles)
  mva_thresholds = mva_quantiles.tolist()
  return mva_thresholds

def calculate_significance(root_filename, tree_name, branches, fixed_width=False):
  signal_fraction_edges = [0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]

  tmva_chain = ROOT.TChain(tree_name)
  tmva_chain.Add(root_filename)
  mva_threshold_edges = find_signal_fraction_thresholds(signal_fraction_edges, tmva_chain, branches['observable'], branches['yhat'], branches['y'], branches['weight'])
  # Add minimum and maximum
  mva_threshold_edges.insert(0,tmva_chain.GetMinimum(branches['yhat']))
  mva_threshold_edges.append(tmva_chain.GetMaximum(branches['yhat']))
  #print(mva_threshold_edges)

  # signal_widths = [ [90%(low),65%(low),65%(high),90%(high)] ]
  signal_widths = []
  # significances = []
  significances = []

  hist = ROOT.TH1F("hist","hist",160,100,180)
  for ithresh, mva_threshold in enumerate(mva_threshold_edges):
    # Find width of signals for MVA threshold bin
    if ithresh == 0: continue
    if ithresh == 1: mva_window = f'{branches["yhat"]}<{mva_threshold}'
    elif ithresh == len(mva_threshold_edges)-1: mva_window = f'{branches["yhat"]}>{mva_threshold_edges[ithresh-1]}'
    else: mva_window = f'{branches["yhat"]}<{mva_threshold}&&{branches["yhat"]}>{mva_threshold_edges[ithresh-1]}'
    entries = tmva_chain.Draw(f'{branches["observable"]}>>hist',f'({branches["y"]}==1&&{mva_window})*{branches["weight"]}',"goff")
    if fixed_width: signal_widths.append([120, 122, 128, 130])
    else:
      mva_signal_width = hist.GetStdDev()
      mass_fractions = [0.95, 0.84, 0.16, 0.05] # 90%, 65%
      mass_quantiles = array.array('d', [0.]*len(mass_fractions))
      mass_fractions = array.array('d', [1.-mass_fraction for mass_fraction in mass_fractions])
      hist.GetQuantiles(len(mass_fractions), mass_quantiles, mass_fractions)
      mass_thresholds = mass_quantiles.tolist()
      #print(ithresh, mva_window, entries, mva_signal_width)
      #print('  ',mass_thresholds)
      signal_widths.append(mass_thresholds)

    # Find signal and background yield within 2 sigma of signal width
    mass_window = f'{branches["observable"]}<{signal_widths[-1][3]}&&{branches["observable"]}>{signal_widths[-1][0]}' #90% of signal
    #print('  ',mass_window)
    tmva_chain.Draw(f"{branches['observable']}>>hist",f'({branches["y"]}==1&&{mva_window}&&{mass_window})*{branches["weight"]}',"goff")
    nevents_signal = hist.GetSum()
    nentries_signal = hist.GetEntries()
    tmva_chain.Draw(f"{branches['observable']}>>hist",f'({branches["y"]}==0&&{mva_window}&&{mass_window})*{branches["weight"]}',"goff")
    nevents_background = hist.GetSum()
    nentries_background = hist.GetEntries()
    #print(f'  nSig: {nevents_signal} nBkg: {nevents_background}')
    #print(f'  Entries sig: {nentries_signal} bkg: {nentries_background}')
    if nevents_background == 0: significances.append(0)
    else: significances.append(math.sqrt(2*((nevents_signal+nevents_background)*math.log(1+nevents_signal*1./nevents_background)-nevents_signal)))
    #print(f'  Significance: {significances[-1]}')

  # Find integrated significance
  integrated_significance = 0.
  for significance in significances:
    integrated_significance += significance**2
  integrated_significance = math.sqrt(integrated_significance)
  #print(f'  Integrated significance: {integrated_significance:.4f}')

  return integrated_significance, significances, signal_widths

# mva_info = {'train': {'x':, 'y':, 'yhat':, 'observable':, 'weight':}, 'test': {...}, 'name':, 'filename':, names: {'train_tree':, 'test_tree':, 'y':, 'yhat':, 'observable': 'weight':, 'x':}}
def evaluate_significance_bins_with_resolution(info_mvas, draw=True, tree_type='eval_tree', fixed_width = False, paper=False):
  """
  Uses below information
  - mva_info['names']['test_filename', 'train_filename', 'eval_filename', 'test_tree', 'train_tree', 'eval_tree']
  - mva_info['names']['observable', 'y', 'weight', 'yhat']
  - mva_info['name']
  """
  verbose = False
  #verbose = True
  #signal_fractions = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1] #Last is due to binning code below
  signal_fractions = [0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.125] #Last is due to binning code below
  #signal_fractions = [0.8, 0.6, 0.4, 0.2, 0.2] #Last is due to binning code below
  #signal_fractions = [0.6, 0.3, 0.3] #Last is due to binning code below
  # Draw how significance changes for each mva
  np_x = np.array(signal_fractions)
  np_x[-1] = 0. # Last bin is special
  if len(signal_fractions) == 5: np_x += 0.1 # For 5 bins
  elif len(signal_fractions) == 10: np_x += 0.05 # For 10 bins
  else: np_x += 0.0625 # For 8 bins
  #np_x += 0.1 # Last bin is special # For 5 bins
  #x_array = array.array('d', signal_fractions)
  x_array = array.array('d', np_x)
  # significance_arrays[mva_name] = [signficance, ...]
  significance_arrays = {}
  significance_squared_arrays = {}
  purity_arrays = {}
  signal_arrays = {}
  background_arrays = {}
  resolution_arrays = {}
  resolution_windows = {}
  significance_graphs = {}
  purity_graphs = {}
  signal_graphs = {}
  background_graphs = {}
  resolution_graphs = {}
  best_significances = []
  integrated_significances= []
  purity_sigeff70s = [0.]
  significance_min_max = []
  purity_min_max = []
  resolution_min_max = []
  signal_min_max = []
  background_min_max = []
  for mva_info in info_mvas:
    # Set names
    tree_name = mva_info['names'][tree_type]
    if tree_type == 'train_tree': tmva_filename = mva_info['names']['train_filename']
    elif tree_type == 'eval_tree': tmva_filename = mva_info['names']['eval_filename']
    else: tmva_filename = mva_info['names']['test_filename']
    mass_name = mva_info['names']['observable']
    label_name = mva_info['names']['y']
    weight_name = mva_info['names']['weight']
    mva_name = mva_info['names']['yhat']
    mva_tag = mva_info['name']
    # Make arrays
    significance_arrays[mva_tag] = array.array('d')
    significance_squared_arrays[mva_tag] = array.array('d')
    purity_arrays[mva_tag] = array.array('d')
    resolution_arrays[mva_tag] = array.array('d')
    signal_arrays[mva_tag] = array.array('d')
    background_arrays[mva_tag] = array.array('d')

    tmva_chain = ROOT.TChain(tree_name)
    tmva_chain.Add(tmva_filename)
    #luminosity = 137. + 110.
    luminosity = 1.
    hist = ROOT.TH1F("hist","hist",160,100,180)
    # Find mva thresholds
    mva_thresholds = find_signal_fraction_thresholds(signal_fractions, tmva_chain, mass_name, mva_name, label_name, weight_name)
    #sigeff70_mva_threshold = mva_thresholds[signal_fractions.index(0.7)]
    if verbose: print(f'signal_fractions: {signal_fractions} mva_thresholds: {mva_thresholds}')
    # mva_signal_width[threshold] = signal_width
    mva_signal_widths = {}
    # Find width of signals for MVA threshold bin
    for ithresh, mva_threshold in enumerate(mva_thresholds):
      if ithresh == 0:
        mva_window = f'{mva_name}<{mva_threshold}'
      elif ithresh == len(mva_thresholds)-1:
        mva_window = f'{mva_name}>{mva_threshold}'
      else:
        mva_window = f'{mva_name}<{mva_threshold}&&{mva_name}>{mva_thresholds[ithresh-1]}'
      entries = tmva_chain.Draw(mass_name+">>hist",f'({label_name}==1&&{mva_window})*{weight_name}',"goff")
      mva_signal_width = hist.GetStdDev()
      mva_signal_widths[mva_threshold] = mva_signal_width
      #resolution_arrays[mva_tag].append(mva_signal_width)
      #mass_fractions = [0.975, 0.84, 0.16, 0.025]
      mass_fractions = [0.95, 0.84, 0.16, 0.05] # 90%, 65%
      mass_quantiles = array.array('d', [0.]*len(mass_fractions))
      mass_fractions = array.array('d', [1.-mass_fraction for mass_fraction in mass_fractions])
      hist.GetQuantiles(len(mass_fractions), mass_quantiles, mass_fractions)
      mass_thresholds = mass_quantiles.tolist()
      if verbose: print(f'mass thresh: {mass_thresholds}')
      if verbose: print(f'90 window: {mass_thresholds[3]-mass_thresholds[0]} 68 window: {mass_thresholds[2]-mass_thresholds[1]}')
      resolution_arrays[mva_tag].append(mass_thresholds[2]-mass_thresholds[1]) # 65% of signal
      if mva_tag not in resolution_windows: resolution_windows[mva_tag] = {}
      resolution_windows[mva_tag][mva_threshold] = mass_thresholds
      if verbose: print("mva window: "+str(mva_window)+" signal_width: "+str(resolution_arrays[mva_tag][-1])+" entries: "+str(entries))
    # Find signal and background within 2 sigma of signal width
    significances = {}
    purities = {}
    sigma = 2
    # Find total signal and background events
    mva_signal_width = 2.5
    mass_window = f'{mass_name}<125+{mva_signal_width*sigma}&&{mass_name}>(125-{mva_signal_width*sigma})'
    tmva_chain.Draw(mass_name+">>hist",f'({label_name}==1&&{mass_window})*{weight_name}',"goff")
    nevents_signal = hist.GetSum()
    tmva_chain.Draw(mass_name+">>hist",f'({label_name}==0&&{mass_window})*{weight_name}',"goff")
    nevents_background = hist.GetSum()
    #if verbose: print(f'All nSig: {nevents_signal} nBkg: {nevents_background} signi: {nevents_signal/math.sqrt(nevents_background)}')
    if verbose: print(f'All nSig: {nevents_signal} nBkg: {nevents_background} signi: {math.sqrt(2*((nevents_signal+nevents_background)*math.log(1+nevents_signal*1./nevents_background)-nevents_signal))}')
    for ithresh, mva_threshold in enumerate(mva_thresholds):
      mva_signal_width = mva_signal_widths[mva_threshold]
      if fixed_width: 
        #mva_signal_width = 2.
        mass_window = f'{mass_name}<130&&{mass_name}>120' 
      else:
        #mass_window = f'{mass_name}<125+{mva_signal_width*sigma}&&{mass_name}>(125-{mva_signal_width*sigma})'
        mass_window = f'{mass_name}<{resolution_windows[mva_tag][mva_threshold][3]}&&{mass_name}>{resolution_windows[mva_tag][mva_threshold][0]}' #90% of signal
        #mass_window = f'{mass_name}<{resolution_windows[mva_tag][mva_threshold][2]}&&{mass_name}>{resolution_windows[mva_tag][mva_threshold][1]}' #65% of signal
      if ithresh == 0:
        mva_window = f'{mva_name}<{mva_threshold}'
      elif ithresh == len(mva_thresholds)-1:
        mva_window = f'{mva_name}>{mva_threshold}'
      else:
        mva_window = f'{mva_name}<{mva_threshold}&&{mva_name}>{mva_thresholds[ithresh-1]}'
      tmva_chain.Draw(mass_name+">>hist",f'({label_name}==1&&{mva_window}&&{mass_window})*{weight_name}*{luminosity}',"goff")
      nevents_signal = hist.GetSum()
      nentries_signal = hist.GetEntries()
      tmva_chain.Draw(mass_name+">>hist",f'({label_name}==0&&{mva_window}&&{mass_window})*{weight_name}*{luminosity}',"goff")
      nevents_background = hist.GetSum()
      nentries_background = hist.GetEntries()
      #if verbose: print(f'signal width: {mva_signal_width} sigma: {sigma} mva_threshold: {mva_window} nSig: {nevents_signal} nBkg: {nevents_background}')
      if verbose: print(f'mva_threshold: {mva_window} nSig: {nevents_signal} nBkg: {nevents_background}')
      if verbose: print(f'                     mass window: {mass_window}')
      if verbose: print(f'                     Entries sig: {nentries_signal} bkg: {nentries_background}')
      # Calculate significance
      if nevents_background != 0:
        #if verbose: print(f"mva_threshold: {mva_threshold:.4f} significance [s/sqrt(b)]: {nevents_signal/math.sqrt(nevents_background):.4f}")
        if verbose: print(f"  mva_threshold: {mva_threshold:.4f} significance [s/sqrt(b)]: {math.sqrt(2*((nevents_signal+nevents_background)*math.log(1+nevents_signal*1./nevents_background)-nevents_signal)):.4f}")
        #significances[mva_threshold] = nevents_signal/math.sqrt(nevents_background)
        #significance_arrays[mva_tag].append(nevents_signal/math.sqrt(nevents_background))
        #significance_squared_arrays[mva_tag].append(nevents_signal**2/nevents_background)
        significances[mva_threshold] = math.sqrt(2*((nevents_signal+nevents_background)*math.log(1+nevents_signal*1./nevents_background)-nevents_signal))
        significance_arrays[mva_tag].append(math.sqrt(2*((nevents_signal+nevents_background)*math.log(1+nevents_signal*1./nevents_background)-nevents_signal)))
        significance_squared_arrays[mva_tag].append(2*((nevents_signal+nevents_background)*math.log(1+nevents_signal*1./nevents_background)-nevents_signal))
      else: 
        print(f'[Info] mva_tag: {mva_tag} mva_threshold: {mva_threshold:.4f} does not have background events. Setting significance to 0.')
        significances[mva_threshold] = 0
        significance_arrays[mva_tag].append(0)
        significance_squared_arrays[mva_tag].append(0)
      ## Calculate purity
      #if nevents_background+nevents_signal !=0:
      #  if verbose: print(f"  mva_threshold: {mva_threshold:.4f} purity (s/(b+s)): {nevents_signal/(nevents_background+nevents_signal):.4f}")
      #  purity_arrays[mva_tag].append(nevents_signal/(nevents_background+nevents_signal))
      #else: 
      #  print(f'[Info] mva_tag: {mva_tag} mva_threshold: {mva_threshold:.4f} does not have signal+background events. Setting purity to 0.')
      #  purity_arrays[mva_tag].append(0)
      #if mva_threshold == sigeff70_mva_threshold:
      #  purity_sigeff70 = purity_arrays[mva_tag][-1]
      signal_arrays[mva_tag].append(nevents_signal)
      background_arrays[mva_tag].append(nevents_background)
    # Find best significance
    best_mva = -999
    best_significance = -999
    for mva_threshold in mva_thresholds:
      if significances[mva_threshold] > best_significance:
        best_significance = significances[mva_threshold]
        best_mva = mva_threshold
    #print(f'{mva_tag} Best mva threshold: {best_mva:.4f} significance: {best_significance:.4f}')
    # Find integrated significance
    integrated_significance = 0.
    for significance in significance_arrays[mva_tag]:
      integrated_significance += significance**2
    integrated_significance = math.sqrt(integrated_significance)
    integrated_significances.append(integrated_significance)
    print(f'{mva_tag} Integrated significance: {integrated_significance:.4f}')
    best_significances.append(best_significance)
    #purity_sigeff70s.append(purity_sigeff70)
    # Make graph
    #significance_graph = ROOT.TGraph(len(signal_fractions), x_array, significance_squared_arrays[mva_tag])
    significance_graph = ROOT.TGraph(len(signal_fractions), x_array, significance_arrays[mva_tag])
    significance_graphs[mva_tag] = significance_graph
    #purity_graph = ROOT.TGraph(len(signal_fractions), x_array, purity_arrays[mva_tag])
    #purity_graphs[mva_tag] = purity_graph
    resolution_graph = ROOT.TGraph(len(signal_fractions), x_array, resolution_arrays[mva_tag])
    resolution_graphs[mva_tag] = resolution_graph
    signal_graph = ROOT.TGraph(len(signal_fractions), x_array, signal_arrays[mva_tag])
    signal_graphs[mva_tag] = signal_graph
    background_graph = ROOT.TGraph(len(signal_fractions), x_array, background_arrays[mva_tag])
    background_graphs[mva_tag] = background_graph
    # Find min max of graph
    x_min, x_max, y_min, y_max = ctypes.c_double(), ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
    significance_graph.ComputeRange(x_min, y_min, x_max, y_max)
    if len(significance_min_max) == 0: significance_min_max = [y_min.value, y_max.value]
    else: 
      if y_min.value < significance_min_max[0]: significance_min_max[0] = y_min.value
      if y_max.value > significance_min_max[1]: significance_min_max[1] = y_max.value
    #print(f'y_min: {y_min}, y_max: {y_max}, min_max[0]: {significance_min_max[0]}, min_max[1]: {significance_min_max[1]}')
    #purity_graph.ComputeRange(x_min, y_min, x_max, y_max)
    #if len(purity_min_max) == 0: purity_min_max = [y_min.value, y_max.value]
    #else: 
    #  if y_min.value < purity_min_max[0]: purity_min_max[0] = y_min.value
    #  if y_max.value > purity_min_max[1]: purity_min_max[1] = y_max.value
    resolution_graph.ComputeRange(x_min, y_min, x_max, y_max)
    if len(resolution_min_max) == 0: resolution_min_max = [y_min.value, y_max.value]
    else: 
      if y_min.value < resolution_min_max[0]: resolution_min_max[0] = y_min.value
      if y_max.value > resolution_min_max[1]: resolution_min_max[1] = y_max.value
    signal_graph.ComputeRange(x_min, y_min, x_max, y_max)
    if len(signal_min_max) == 0: signal_min_max = [y_min.value, y_max.value]
    else: 
      if y_min.value < signal_min_max[0]: signal_min_max[0] = y_min.value
      if y_max.value > signal_min_max[1]: signal_min_max[1] = y_max.value
    background_graph.ComputeRange(x_min, y_min, x_max, y_max)
    if len(background_min_max) == 0: background_min_max = [y_min.value, y_max.value]
    else: 
      if y_min.value < background_min_max[0]: background_min_max[0] = y_min.value
      if y_max.value > background_min_max[1]: background_min_max[1] = y_max.value
    #print(f'y_min: {y_min}, y_max: {y_max}, min_max[0]: {purity_min_max[0]}, min_max[1]: {purity_min_max[1]}')
  # Draw graphs
  if draw:
    if paper:
      significance_legend = ROOT.TLegend(0.2, 0.60, 0.7, 0.87)
    else:
      significance_legend = ROOT.TLegend(0.5, 0.6, 0.9, 0.98)
    colors = [4,800-7,8,1,6,7,8,9,46,49,41,38,30,33]
    if len(significance_graphs) >= 15: 
      print('Need to add more colors to draw everything')
    else:
      c1 = new_canvas()
      c1.SetLeftMargin(0.15)
      for iMva, mva_tag in enumerate(significance_graphs):
        #significance_graphs[mva_tag].SetTitle(mva_tag)
        integrated_significance = integrated_significances[iMva]
        significance_graphs[mva_tag].SetTitle(f';Signal percentile;Bin significance')
        if paper:
          #significance_graphs[mva_tag].GetXaxis().SetLimits(0.,1.) # x range
          significance_graphs[mva_tag].GetHistogram().SetMaximum(1.5) # y range
          significance_graphs[mva_tag].GetHistogram().SetMinimum(0.)
          name = re.sub('_s[1-3]', '',mva_tag)
          significance_legend.AddEntry(significance_graphs[mva_tag], f'{name}, Signi. = {integrated_significance:.2f}')
          if iMva == 0: 
              significance_graphs[mva_tag].Draw("APL RX")
          else: significance_graphs[mva_tag].Draw("PL RX")
        else:
          significance_graphs[mva_tag].GetYaxis().SetRangeUser(significance_min_max[0]-0.01, significance_min_max[1]+0.01)
          significance_legend.AddEntry(significance_graphs[mva_tag], f'{mva_tag}, comb. signi= {integrated_significance:.3f}')
          if iMva == 0: significance_graphs[mva_tag].Draw("APL same")
          else: significance_graphs[mva_tag].Draw("PL")
        significance_graphs[mva_tag].SetLineColor(colors[iMva])
        significance_graphs[mva_tag].SetMarkerColor(colors[iMva])
        significance_graphs[mva_tag].SetMarkerStyle(21)
        significance_graphs[mva_tag].SetMarkerSize(0.3)
      significance_legend.Draw()
      c1.SaveAs(f'plots/significances_with_resolution_{tree_type}.pdf')

      c3 = new_canvas()
      c3.SetLeftMargin(0.15)
      resolution_legend = ROOT.TLegend(0.2, 0.6, 0.5, 0.98)
      for iMva, mva_tag in enumerate(resolution_graphs):
        resolution_graphs[mva_tag].SetTitle(f';Signal percentile;signal resolution')
        if iMva == 0: resolution_graphs[mva_tag].Draw("APL")
        else: resolution_graphs[mva_tag].Draw("PL")
        resolution_graphs[mva_tag].SetLineColor(colors[iMva])
        resolution_graphs[mva_tag].SetMarkerColor(colors[iMva])
        resolution_graphs[mva_tag].SetMarkerStyle(21)
        resolution_graphs[mva_tag].SetMarkerSize(0.3)
        resolution_graphs[mva_tag].GetYaxis().SetRangeUser(resolution_min_max[0]-0.1, resolution_min_max[1]+0.1)
        resolution_legend.AddEntry(resolution_graphs[mva_tag], f'{mva_tag}')
        resolution_legend.Draw()
      c3.SaveAs(f'plots/signal_resolution_{tree_type}.pdf')

      c4 = new_canvas()
      c4.SetLeftMargin(0.15)
      c4.SetLogy()
      background_legend = ROOT.TLegend(0.5, 0.9, 0.9, 0.98)
      for iMva, mva_tag in enumerate(background_graphs):
        background_graphs[mva_tag].SetTitle(f';signal eff.;background entries')
        if iMva == 0: 
          background_graphs[mva_tag].Draw("APL same")
        else: 
          background_graphs[mva_tag].Draw("PL")
        background_graphs[mva_tag].SetLineColor(colors[iMva])
        background_graphs[mva_tag].SetMarkerColor(colors[iMva])
        background_graphs[mva_tag].SetMarkerStyle(21)
        background_graphs[mva_tag].SetMarkerSize(0.3)
        background_graphs[mva_tag].GetYaxis().SetRangeUser(background_min_max[0]*0.5, background_min_max[1]*1.05)
        background_legend.AddEntry(background_graphs[mva_tag], f'{mva_tag}')
        background_legend.Draw()
      c4.SaveAs(f'plots/background_distribution_{tree_type}.pdf')

  return integrated_significances, purity_sigeff70s

def calculate_roc(root_filename, tree_name, branches, narrow_mass_window=False):
  root_file = uproot.open(root_filename)
  y_array = root_file[tree_name][branches['y']].array(library='np')
  yhat_array = root_file[tree_name][branches['yhat']].array(library='np')
  weight_array = root_file[tree_name][branches['weight']].array(library='np')
  observable_array = root_file[tree_name][branches['observable']].array(library='np')
  print('Setting negative weights to zero for roc calculation')
  weight_array[weight_array<0] = 0.
  if narrow_mass_window:
    # Cut on mass
    mass_cut = (observable_array<130).__and__(observable_array>120)
    y_array = y_array[mass_cut]
    yhat_array = yhat_array[mass_cut]
    weight_array = weight_array[mass_cut]
    fpr, tpr, threshold = sklearn.metrics.roc_curve(y_array, yhat_array, sample_weight=weight_array)
  else:
    fpr, tpr, threshold = sklearn.metrics.roc_curve(y_data, yhat_data, sample_weight=weight_array)
  auc = sklearn.metrics.auc(fpr, tpr)*100
  return auc, fpr, tpr

def calculate_correlation():
 

def evaluate_roc(info_mvas, no_weight=False, full_mass_window=True, dataset='eval', paper=False):
  """
  Uses below information
  - mva_info[dataset]['y', 'yhat', 'weight', 'observable']
  - mva_info['name']
  """
  roc_infos = []
  # mva_info = {'train': {'x':, 'y':, 'yhat':, 'observable':, 'weight':}, 'test': {...}, 'name':}
  for mva_info in info_mvas:
    # Set information
    mva_name = mva_info['name']
    y_data = mva_info[dataset]['y']
    yhat_data = mva_info[dataset]['yhat']
    weight_data = mva_info[dataset]['weight']
    observable_data = mva_info[dataset]['observable']
    # Evaluate with ROC curve
    if no_weight:
      # roc with no weight
      fpr, tpr, threshold = sklearn.metrics.roc_curve(y_dat, yhat_data)
    else:
      # roc with weights probably not working due to negative weights.
      sample_weight = np.array(weight_data)
      sample_weight[sample_weight<0] = 0.
      if full_mass_window:
        fpr, tpr, threshold = sklearn.metrics.roc_curve(y_data, yhat_data, sample_weight=sample_weight)
      else:
        # Cut on mass
        mass = np.array(observable_data)
        mass_cut = (mass<130).__and__(mass>120)
        y_array = np.array(y_data)[mass_cut]
        yhat_array = np.array(yhat_data)[mass_cut]
        sample_weight = sample_weight[mass_cut]
        fpr, tpr, threshold = sklearn.metrics.roc_curve(y_array, yhat_array, sample_weight=sample_weight)
    roc_infos.append([fpr, tpr, threshold, mva_name])
    #print(f'fpr: {fpr}\ntpr: {tpr}\nthresh: {threshold}')

  plt.figure(figsize=(5,5))
  for roc_info in roc_infos:
    fpr, tpr, threshold, name = roc_info
    print(f'{name} : AUC: {sklearn.metrics.auc(fpr, tpr)*100}')
  for roc_info in roc_infos:
    fpr, tpr, threshold, name = roc_info
    if paper:
      name = re.sub('_s[1-3]', '', name)
      if 'Random' in name:
        plt.plot(fpr, tpr, lw=1, label=name+", AUC = {:.1f}%".format(sklearn.metrics.auc(fpr, tpr)*100), color='black')
      else:
        plt.plot(fpr, tpr, lw=1, label=name+", AUC = {:.1f}%".format(sklearn.metrics.auc(fpr, tpr)*100))
    else:
      plt.plot(fpr, tpr, lw=1, label=name+", AUC = {:.1f}%".format(sklearn.metrics.auc(fpr, tpr)*100))
  #plt.plot([0.,1.],[0.,1.], label='Random classifier, AUC = 50.0%', linestyle='dashed', color='black')
  plt.xlabel(r'False positive rate')
  plt.ylabel(r'True positive rate')
  #plt.title(r'Receiver Operating Characteristic (ROC)')
  #plt.semilogy()
  plt.ylim(0,1)
  plt.xlim(0,1)
  #plt.grid(True)
  plt.legend(loc='lower right')
  if no_weight: filename = "plots/roc_noweight_higgsToZGamma_classifiers.pdf"
  else: filename = "plots/roc_higgsToZGamma_classifiers.pdf"
  plt.savefig(filename)
  print(f"Saved to {filename}")

# mva_info = {'train': {'x':, 'y':, 'yhat':, 'observable':, 'weight':}, 'test': {...}, 'name':, 'filename':, names: {'train_tree':, 'test_tree':, 'y':, 'yhat':, 'observable': 'weight':, 'x':}}
def evaluate_correlation(info_mvas, draw=True, tree_type='eval_tree', paper=False):
  """
  Uses below information
  - mva_info['names']['observable', 'y', 'yhat', 'weight']
  - mva_info['names']['train_filename', 'eval_filename', 'test_filename', 'train_tree', 'eval_tree', 'test_tree']
  - mva_info['name']
  """
  verbose = False
  std_divs = []
  for mva_info in info_mvas:
    if tree_type == 'train_tree': tmva_filename = mva_info['names']['train_filename']
    elif tree_type == 'eval_tree': tmva_filename = mva_info['names']['eval_filename']
    else: tmva_filename = mva_info['names']['test_filename']
    tree_name = mva_info['names'][tree_type]
    mass_name = mva_info['names']['observable']
    label_name = mva_info['names']['y']
    mva_name = mva_info['names']['yhat']
    mva_tag = mva_info['name']
    weight_name = mva_info['names']['weight']

    tmva_chain = ROOT.TChain(tree_name)
    tmva_chain.Add(tmva_filename)
    # Find mva thresholds
    #signal_fractions = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    signal_fractions = [0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]
    #signal_fractions = [0.8, 0.6, 0.4, 0.2]
    mva_thresholds = find_signal_fraction_thresholds(signal_fractions, tmva_chain, mass_name, mva_name, label_name, weight_name)
    mva_max = tmva_chain.GetMaximum(mva_name)
    mva_thresholds.append(mva_max) # Add last bin
    # Make mass histograms with thresholds
    mva_hists = {}
    signal_hists = {}
    ROOT.gStyle.SetOptStat(0)
    #print(f'mva_thresholds: {mva_thresholds}')
    # Make background and signal hists
    for imva, mva_threshold in enumerate(mva_thresholds):
      # Background hists
      hist = ROOT.TH1F("hist_"+str(mva_threshold),"hist_"+str(mva_threshold),80,100,180)
      if imva == 0:
        entries = tmva_chain.Draw(mass_name+">>hist_"+str(mva_threshold),f'{label_name}==0&&{mva_name}<{mva_threshold}',"goff")
        #print(f'{label_name}==0&&{mva_name}<{mva_threshold}, {entries}')
      elif imva == len(mva_thresholds)-1:
        entries = tmva_chain.Draw(mass_name+">>hist_"+str(mva_threshold),f'{label_name}==0&&{mva_name}>{prev_mva_threshold}',"goff")
        #print(f'{label_name}==0&&{mva_name}>{prev_mva_threshold}, {entries}')
      else:
        entries = tmva_chain.Draw(mass_name+">>hist_"+str(mva_threshold),f'{label_name}==0&&{mva_name}<{mva_threshold}&&{mva_name}>{prev_mva_threshold}',"goff")
        #print(f'{label_name}==0&&{mva_name}<{mva_threshold}&&{mva_name}>{prev_mva_threshold}, {entries}')
      # Normalize histogram
      sum_weight = hist.GetSumOfWeights()
      if sum_weight != 0:
        hist.Scale(1/sum_weight)
        mva_hists[mva_threshold] = hist

      # Signal hists
      signal_hist = ROOT.TH1F("signal_hist_"+str(mva_threshold),"signal_hist_"+str(mva_threshold),80,100,180)
      if imva == 0:
        #print(f'{label_name}==1&&{mva_name}<{mva_threshold}')
        entries = tmva_chain.Draw(mass_name+">>signal_hist_"+str(mva_threshold),f'{label_name}==1&&{mva_name}<{mva_threshold}',"goff")
      elif imva == len(mva_thresholds)-1:
        #print(f'{label_name}==1&&{mva_name}>{prev_mva_threshold}')
        entries = tmva_chain.Draw(mass_name+">>signal_hist_"+str(mva_threshold),f'{label_name}==1&&{mva_name}>{mva_threshold}',"goff")
      else:
        #print(f'{label_name}==1&&{mva_name}<{mva_threshold}&&{mva_name}>{prev_mva_threshold}')
        entries = tmva_chain.Draw(mass_name+">>signal_hist_"+str(mva_threshold),f'{label_name}==1&&{mva_name}<{mva_threshold}&&{mva_name}>{prev_mva_threshold}',"goff")
      prev_mva_threshold = mva_threshold
      # Normalize histogram
      sum_weight = signal_hist.GetSumOfWeights()
      if sum_weight != 0:
        signal_hist.Scale(1/sum_weight)
        signal_hists[mva_threshold] = signal_hist

    # Find std_dev of histograms
    # Get values of histogram bin. 1 is first bin, nbin is last bin
    #print(mva_hists[mva_thresholds[0]].GetNbinsX())
    std_values = []
    #for iBin in range(mva_hists[mva_thresholds[0]].GetNbinsX()):
    for iBin in range(next(iter(mva_hists.values())).GetNbinsX()):
      bin_values = []
      for mva_threshold in mva_hists:
        hist_ibin_entry = mva_hists[mva_threshold].GetBinContent(iBin+1)
        bin_values.append(hist_ibin_entry)
        #print("iBin: "+str(iBin+1)+" mva_threshold: "+str(mva_threshold)+" "+str(hist_ibin_entry))
      stdev = np.std(bin_values, dtype=np.float64)
      mean = np.mean(bin_values, dtype=np.float64)
      if (stdev == 0 and mean == 0): continue # Ignore case where hist bins are empty
      #if verbose: print("iBin: "+str(iBin+1)+" x: "+str(mva_hists[mva_thresholds[0]].GetBinCenter(iBin+1))+" stdev: "+str(stdev))
      if verbose: print("iBin: "+str(iBin+1)+" x: "+str(next(iter(mva_hists.values())).GetBinCenter(iBin+1))+" stdev: "+str(stdev))
      std_values.append(stdev)
    std_divs.append(np.mean(std_values))
    print(f'Correlation: {mva_tag} mean stdev for bkg: {np.mean(std_values)}')

    if draw:
      # background
      c1 = new_canvas()
      if paper:
        legend = ROOT.TLegend(0.6, 0.5, 0.90, 0.88)
      else:
        legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.98)
      if paper:
        bkg_hist = ROOT.TH1F("bkg_hist",";m_{ll#gamma} [GeV];Normalized / 1 GeV",80,100,180)
        bkg_hist.GetXaxis().SetTitleSize(0.07)
        bkg_hist.GetXaxis().SetLabelSize(0.07)
        bkg_hist.GetYaxis().SetTitleSize(0.07)
        bkg_hist.GetYaxis().SetLabelSize(0.07)
        bkg_hist.GetXaxis().SetLabelOffset(0.02)
        bkg_hist.GetXaxis().SetTitleOffset(1.2)
        bkg_hist.SetMinimum(0.)
        bkg_hist.SetMaximum(0.05)
        bkg_hist.GetXaxis().SetNdivisions(4,4,0,False)
        bkg_hist.GetYaxis().SetNdivisions(5,4,0,False)
        #ROOT.gStyle.SetTitleY(.97)
        #c1.SetLeftMargin(0.15)
        c1.SetTopMargin(0.1)
        c1.SetRightMargin(0.08)
        c1.SetLeftMargin(0.28)
        c1.SetBottomMargin(0.18)
      else:
        bkg_hist = ROOT.TH1F("bkg_hist","Background dist.;m_{llg} [GeV]",160,100,180)
      bkg_hist.Draw()
      #colors = [ROOT.kGreen-8, ROOT.kGreen-5, ROOT.kGreen+4,
      #          ROOT.kBlue-10, ROOT.kBlue-7, ROOT.kBlue-3, ROOT.kBlue,
      #          ROOT.kRed-10, ROOT.kRed-7, ROOT.kRed,]
      colors = [ROOT.kGreen-8, ROOT.kGreen-5, ROOT.kGreen+4,
                ROOT.kBlue-10, ROOT.kBlue-7, 
                ROOT.kRed-10, ROOT.kRed-7, ROOT.kRed,]
      #colors = [ROOT.kGreen-8, ROOT.kGreen+4,
      #          ROOT.kBlue,
      #          ROOT.kRed-10, ROOT.kRed,]
      for imva, mva_threshold in enumerate(mva_hists):
        hist = mva_hists[mva_threshold]
        hist.SetLineColor(colors[imva])
        hist.Draw('same')
        if paper:
          if imva == len(mva_hists)-1: legend.AddEntry(hist, f'sig. p = [{signal_fractions[imva-1]:.3f}, {signal_fractions[imva-1]-1./(len(signal_fractions)+1):.3f}]', 'l')
          else: legend.AddEntry(hist, f'sig. p = [{signal_fractions[imva]:.3f}, {signal_fractions[imva]+1./(len(signal_fractions)+1):.3f}]', 'l')
        else:
          legend.AddEntry(hist)
      legend.Draw()
      box = ROOT.TPaveText(0.32, 0.15, 0.67, 0.35, 'NDC NB')
      box.SetFillColorAlpha(0,0)
      box.AddText(f'Correlation: {np.mean(std_values):.4f}')
      box.Draw()
      #set_max_th1(1.10)
      c1.SaveAs("plots/bkg_hist_mva_"+slugify(mva_tag)+".pdf")

      # signal
      c2 = new_canvas()
      legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.98)
      signal_hist = ROOT.TH1F("signal_hist","Signal dist.;m_{llg} [GeV]",160,100,180)
      signal_hist.Draw('hist')
      for imva, mva_threshold in enumerate(signal_hists):
        hist = signal_hists[mva_threshold]
        hist.SetLineColor(colors[imva])
        hist.Draw('same hist')
        legend.AddEntry(hist)
      legend.Draw()
      set_max_th1()
      c2.SaveAs("plots/signal_hist_mva_"+slugify(mva_tag)+".pdf")
      ROOT.gStyle.SetOptStat(1)

  return std_divs

# Make plot of mva variable for signal/backgroundxtrain/test
def evaluate_overtraining(info_mvas, full_mass_window = True, paper=False):
  """
  Uses below information
  - mva_info['names']['weight', 'observable', 'yhat', 'y']
  - mva_info['names']['test_filename', 'train_filename', 'test_tree', 'train_tree']
  - mva_info['name']
  """
  use_weight = True
  nbins = 100
  # mva_info = {'train': {'x':, 'y':, 'yhat':, 'observable':, 'weight':}, 'test': {...}, 'name':, 'filename':, names: {'train_tree':, 'test_tree':, 'label':, 'yhat':, 'observable': 'weight':, 'x':}}
  for mva_info in info_mvas:
    # Set names
    weight_name = mva_info['names']['weight']
    observable_name = mva_info['names']['observable']
    y_name = mva_info['names']['y']
    yhat_name = mva_info['names']['yhat']
    test_filename = mva_info['names']['test_filename']
    train_filename = mva_info['names']['train_filename']
    test_tree_name = mva_info['names']['test_tree']
    train_tree_name = mva_info['names']['train_tree']
    mva_name = mva_info['name']
    # Make cuts
    cuts = '1'
    if full_mass_window == False: cuts += f"&&{observable_name}>120&&{observable_name}<130"
    # Get trees
    train_root_file = ROOT.TFile(train_filename)
    train_tree = train_root_file.Get(train_tree_name)
    test_root_file = ROOT.TFile(test_filename)
    test_tree = test_root_file.Get(test_tree_name)
    # Find min max of mva variable
    mva_max = max(train_tree.GetMaximum(yhat_name), test_tree.GetMaximum(yhat_name))
    mva_min = min(train_tree.GetMinimum(yhat_name), test_tree.GetMinimum(yhat_name))
    # Test tree
    yhat_signal_test = ROOT.TH1F('signal_test', 'signal_test;mva', nbins, mva_min, mva_max)
    yhat_bkg_test = ROOT.TH1F('bkg_test', 'bkg_test', nbins, mva_min, mva_max)
    if use_weight:
      test_tree.Draw(yhat_name+'>>signal_test', f"(({y_name}==1)&&{cuts})*{weight_name}", 'goff')
      test_tree.Draw(yhat_name+'>>bkg_test', f"(({y_name}==0)&&{cuts})*{weight_name}", 'goff')
    else:
      test_tree.Draw(yhat_name+'>>signal_test', f"({y_name}==1)&&{cuts}", 'goff')
      test_tree.Draw(yhat_name+'>>bkg_test', f"({y_name}==0)&&{cuts}", 'goff')
    # Train tree
    yhat_signal_train = ROOT.TH1F('signal_train', 'signal_train', nbins, mva_min, mva_max)
    yhat_bkg_train = ROOT.TH1F('bkg_train', 'bkg_train', nbins, mva_min, mva_max)
    if use_weight:
      train_tree.Draw(yhat_name+'>>signal_train', f"(({y_name}==1)&&{cuts})*{weight_name}", 'goff')
      train_tree.Draw(yhat_name+'>>bkg_train', f"(({y_name}==0)&&{cuts})*{weight_name}", 'goff')
    else:
      train_tree.Draw(yhat_name+'>>signal_train', f"({y_name}==1)&&{cuts}", 'goff')
      train_tree.Draw(yhat_name+'>>bkg_train', f"({y_name}==0)&&{cuts}", 'goff')
    # Normalize
    normalize_hist(yhat_signal_test)
    normalize_hist(yhat_bkg_test)
    normalize_hist(yhat_signal_train)
    normalize_hist(yhat_bkg_train)

    # Chi2test
    residual_y_signal = array.array('d',[0.]*nbins)
    residual_y_bkg = array.array('d',[0.]*nbins)
    pvalue_signal = yhat_signal_train.Chi2Test(yhat_signal_test, 'WW', residual_y_signal)
    pvalue_bkg = yhat_bkg_train.Chi2Test(yhat_bkg_test, 'WW', residual_y_bkg)

    y_max = np.amax([yhat_signal_test.GetMaximum(), yhat_signal_train.GetMaximum(), yhat_bkg_test.GetMaximum(), yhat_bkg_train.GetMaximum()])
    if paper:
      name = re.sub('_s[1-3]','',mva_name)
      blank_hist = ROOT.TH1F('blank_hist','', nbins, mva_min, mva_max)
    else:
      blank_hist = ROOT.TH1F('blank_hist','Overtraining '+mva_name, nbins, mva_min, mva_max)
    blank_hist.SetMaximum(y_max * 1.1)
    blank_hist.SetMinimum(0.)

    c1 = new_canvas()
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
    #ROOT.gStyle.SetTitleY(.97)
    blank_hist.GetXaxis().SetTitle("Network response")
    blank_hist.GetYaxis().SetTitle("Normalized")
    blank_hist.GetXaxis().SetTitleSize(0.09)
    blank_hist.GetXaxis().SetLabelSize(0.09)
    blank_hist.GetYaxis().SetTitleSize(0.09)
    blank_hist.GetYaxis().SetLabelSize(0.09)
    blank_hist.GetXaxis().SetLabelOffset(0.02)
    blank_hist.GetXaxis().SetTitleOffset(1.2)
    #blank_hist.GetXaxis().SetLimits(0,1.)
    blank_hist.SetMinimum(0.)
    #blank_hist.SetMaximum(0.30)
    blank_hist.GetXaxis().SetNdivisions(5,4,0)
    blank_hist.GetYaxis().SetNdivisions(4,4,0)
    #blank_hist.GetYaxis().SetNdivisions(7,4,0,False)
    blank_hist.Draw()
    yhat_bkg_test.SetLineColor(ROOT.kBlue)
    yhat_bkg_test.SetFillColor(ROOT.kBlue)
    yhat_bkg_test.SetFillStyle(3954)
    yhat_bkg_test.Draw('same hist')
    #yhat_bkg_train.SetLineColor(ROOT.kBlack)
    yhat_bkg_train.SetMarkerStyle(5)
    yhat_bkg_train.SetMarkerSize(1.)
    yhat_bkg_train.SetMarkerColor(ROOT.kBlue)
    yhat_bkg_train.Draw('same')
    yhat_signal_test.SetLineColor(ROOT.kRed)
    yhat_signal_test.SetFillColor(ROOT.kRed)
    yhat_signal_test.SetFillStyle(3945)
    yhat_signal_test.Draw('hist same')
    #yhat_signal_train.SetLineColor(ROOT.kRed)
    yhat_signal_train.SetMarkerStyle(5)
    yhat_signal_train.SetMarkerSize(1.)
    yhat_signal_train.SetMarkerColor(ROOT.kRed)
    yhat_signal_train.Draw('same P')
    if paper:
      legend = ROOT.TLegend(0.45, 0.6, 0.85, 0.85)
    else:
      legend = ROOT.TLegend(0.7, 0.9, 0.9, 0.98)
    if paper:
      legend.AddEntry(yhat_signal_test,  "Signal (test  sample)", "f")
      legend.AddEntry(yhat_signal_train, "Signal (validation sample)", 'p')
      legend.AddEntry(yhat_bkg_test,     "Background (test  sample)", 'f')
      legend.AddEntry(yhat_bkg_train,    "Background (validation sample)", 'p')
    else:
      legend.AddEntry(yhat_signal_test)
      legend.AddEntry(yhat_signal_train)
      legend.AddEntry(yhat_bkg_test)
      legend.AddEntry(yhat_bkg_train)
    legend.Draw()

    low_pad.cd()
    low_pad.SetTopMargin(0.1)
    low_pad.SetRightMargin(0.08)
    low_pad.SetLeftMargin(0.24)
    low_pad.SetBottomMargin(0.22)
    n_bins = blank_hist.GetNbinsX()
    #x_min = blank_hist.GetXaxis().GetBinLowEdge(blank_hist.GetXaxis().GetFirst())
    #x_max = blank_hist.GetXaxis().GetBinUpEdge(blank_hist.GetXaxis().GetLast())
    residual_x = array.array('d',[0.]*n_bins)
    for index in range(n_bins):
      residual_x[index] = mva_min + (mva_max-mva_min)/n_bins * index + (mva_max-mva_min)/n_bins/2
    ROOT.gErrorIgnoreLevel = ROOT.kError
    #print(mva_name+f' pvalue signal: {pvalue_signal} bkg: {pvalue_bkg}')
    ROOT.gErrorIgnoreLevel = ROOT.kPrint
    #print('pvalue_signal: '+str(pvalue_signal))
    #print('pvalue_bkg: '+str(pvalue_bkg))
    print(f'chi^2 test p-value {mva_name} signal: {pvalue_signal} bkg: {pvalue_bkg}')
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
    #residual_signal.GetXaxis().SetLimits(0,1.)
    residual_signal.GetXaxis().SetNdivisions(5,4,0)
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
    zero_line = ROOT.TLine(0, 0, mva_max, 0)
    zero_line.Draw()
    box = ROOT.TPaveText(0.25, 0.75, 0.90, 0.88, 'NDC NB')
    box.SetFillColorAlpha(0,0)
    box.AddText('#chi^{2} test p-value, signal='+f'{pvalue_signal:.3f} background={pvalue_bkg:.3f}')
    box.Draw()
    c1.SaveAs('plots/overtrain_'+slugify(mva_name)+'.pdf')
    ROOT.gStyle.SetOptStat(1)

    test_root_file.Close()
    train_root_file.Close()

if __name__ == '__main__':
  mva_filename = 'mva_output/tmva_bdt_hig19014_results.root'

  integrated_significance, significances, signal_widths = calculate_significance(mva_filename, 'eval_tree_nocut', 
    branches={'y':'classID', 'yhat':'BDT', 'weight':'weightXyear', 'observable':'lly_m'}, fixed_width=False)
  print(f'Total significance: {integrated_significance}')
  auc, fpr, tpr = calculate_roc(mva_filename, 'eval_tree_nocut', branches={'y':'classID', 'yhat':'BDT', 'weight':'weightXyear', 'observable':'lly_m'}, narrow_mass_window=True)
  print(f'AUC: {auc}')

  # Load root file
  mva_file = uproot.open(mva_filename)
  mva = {'train': {'x': mva_file['train_tree_nocut'].arrays(['y_mva','yl_drmax','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta']), 
                   'y':mva_file['train_tree_nocut']['classID'].array(), 
                   'yhat':mva_file['train_tree_nocut']['BDT'].array(), 
                   'observable':mva_file['train_tree_nocut']['lly_m'].array(), 
                   'weight':mva_file['train_tree_nocut']['weightXyear'].array()
                  },
         'eval': {'x': mva_file['eval_tree_nocut'].arrays(['y_mva','yl_drmax','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta']), 
                   'y':mva_file['eval_tree_nocut']['classID'].array(), 
                   'yhat':mva_file['eval_tree_nocut']['BDT'].array(), 
                   'observable':mva_file['eval_tree_nocut']['lly_m'].array(), 
                   'weight':mva_file['eval_tree_nocut']['weightXyear'].array()
                  },
         'test': {'x': mva_file['test_tree_nocut'].arrays(['y_mva','yl_drmax','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta']), 
                   'y':mva_file['test_tree_nocut']['classID'].array(), 
                   'yhat':mva_file['test_tree_nocut']['BDT'].array(), 
                   'observable':mva_file['test_tree_nocut']['lly_m'].array(), 
                   'weight':mva_file['test_tree_nocut']['weightXyear'].array()
                  },
          'name': 'BDT_hig19014',
          'names': {
            'train_filename': mva_filename, 'test_filename': mva_filename, 'eval_filename': mva_filename,
            'train_tree': 'train_tree_nocut', 'eval_tree': 'eval_tree_nocut', 'test_tree': 'test_tree_nocut',
            'y': 'classID', 'x': ['y_mva','yl_drmax','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta'],
            'yhat': 'BDT', 'observable': 'lly_m', 'weight':'weightXyear'},
        }
  mva_file.close()

  info_mvas = [mva]

  # Measure performance
  evaluate_significance_bins_with_resolution(info_mvas, tree_type='eval_tree', fixed_width=False, paper=True)
  evaluate_roc(info_mvas,full_mass_window=False, dataset='eval', paper=True) # negative weights set to 0
  # TODO: Clean up below code
  evaluate_correlation(info_mvas, tree_type='eval_tree', paper=True)
  evaluate_overtraining(info_mvas, full_mass_window=False, paper=False)
