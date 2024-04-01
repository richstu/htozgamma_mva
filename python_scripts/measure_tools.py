import ROOT
import array
from ctypes import c_double
import math
import numpy as np
import os

import unicodedata
import re
def slugify(string):
  return re.sub(r'[-\s]+', '-',re.sub(r'[^\w\s-]', '',
                  unicodedata.normalize('NFKD', string)).strip())

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

def normalize_hist(hist):
  sum_weight = hist.GetSumOfWeights()
  hist.Scale(1/sum_weight)

def find_sample_fraction_thresholds(sample_fractions, tmva_chain, var_name, sample_cut, weight_name = '', include_min_max=False):
  # Find min max of variable
  min_val = tmva_chain.GetMinimum(var_name)
  max_val = tmva_chain.GetMaximum(var_name)
  #print(f'find signal fraction min: {min_val}, max: {max_val}')
  hist_var = ROOT.TH1F("hist_var","hist_var",10000,min_val*0.9,max_val*1.1)
  if weight_name == '': n_entries = tmva_chain.Draw(var_name+">>hist_var", sample_cut, 'goff')
  else: n_entries = tmva_chain.Draw(var_name+">>hist_var", f'({sample_cut})*{weight_name}', 'goff')
  var_quantiles = array.array('d', [0.]*len(sample_fractions))
  var_fractions = array.array('d', [1.-sample_fraction for sample_fraction in sample_fractions])
  hist_var.GetQuantiles(len(sample_fractions), var_quantiles, var_fractions)
  #print(var_fractions,var_quantiles)
  var_thresholds = var_quantiles.tolist()
  if include_min_max:
    var_thresholds.insert(0,hist_var.GetXaxis().GetXmin())
    var_thresholds.append(hist_var.GetXaxis().GetXmax())
  return var_thresholds

def find_signal_fraction_thresholds(signal_fractions, tmva_chain, mva_name, label_name, weight_name = ''):
  return find_sample_fraction_thresholds(signal_fractions, tmva_chain, mva_name, f'{label_name}==1', weight_name = '')

#def find_signal_fraction_thresholds(signal_fractions, tmva_chain, mva_name, label_name, weight_name = ''):
#  # Find min max of variable
#  min_val = tmva_chain.GetMinimum(mva_name)
#  max_val = tmva_chain.GetMaximum(mva_name)
#  #print(f'find signal fraction min: {min_val}, max: {max_val}')
#  hist_mva = ROOT.TH1F("hist_mva","hist_mva",10000,min_val*0.9,max_val*1.1)
#  if weight_name == '': n_entries = tmva_chain.Draw(mva_name+">>hist_mva", label_name+"==1", 'goff')
#  else: n_entries = tmva_chain.Draw(mva_name+">>hist_mva", f'({label_name}==1)*{weight_name}', 'goff')
#  mva_quantiles = array.array('d', [0.]*len(signal_fractions))
#  mva_fractions = array.array('d', [1.-signal_fraction for signal_fraction in signal_fractions])
#  hist_mva.GetQuantiles(len(signal_fractions), mva_quantiles, mva_fractions)
#  #print(mva_fractions,mva_quantiles)
#  mva_thresholds = mva_quantiles.tolist()
#  return mva_thresholds

# Used branches: {'y', 'yhat', 'weight', 'observable'}
def calculate_significance(root_filename, tree_name, branches, fixed_width=False, detail_output=False):
  signal_fraction_edges = [0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]
  #signal_fraction_edges = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
  #signal_fraction_edges = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
  #bins = 80
  #signal_fraction_edges = [1-(ibin+1)/(bins) for ibin in range(bins-1)]

  tmva_chain = ROOT.TChain(tree_name)
  tmva_chain.Add(root_filename)
  #print(tree_name)
  #tmva_chain.Scan(f'{branches["y"]}:{branches["observable"]}:{branches["yhat"]}:{branches["weight"]}')
  mva_threshold_edges = find_signal_fraction_thresholds(signal_fraction_edges, tmva_chain, branches['yhat'], branches['y'], branches['weight'])
  #print(mva_threshold_edges)
  # Add minimum and maximum
  mva_threshold_edges.insert(0,tmva_chain.GetMinimum(branches['yhat']))
  mva_threshold_edges.append(tmva_chain.GetMaximum(branches['yhat']))
  #print(mva_threshold_edges)

  # Storage for detail plots
  # signal_widths = [ [90%(low),65%(low),65%(high),90%(high)] ]
  signal_widths = []
  significances = []
  significance_errs = []
  signal_fraction_bins = []

  hist = ROOT.TH1F("hist","hist",160,100,180)
  for ithresh, mva_threshold in enumerate(mva_threshold_edges):
    if ithresh == 0: continue
    # Make mva window cut
    if ithresh == 1: mva_window = f'{branches["yhat"]}<{mva_threshold}'
    elif ithresh == len(mva_threshold_edges)-1: mva_window = f'{branches["yhat"]}>{mva_threshold_edges[ithresh-1]}'
    else: mva_window = f'{branches["yhat"]}<{mva_threshold}&&{branches["yhat"]}>{mva_threshold_edges[ithresh-1]}'
    entries = tmva_chain.Draw(f'{branches["observable"]}>>hist',f'({branches["y"]}==1&&{mva_window})*{branches["weight"]}',"goff")
    # Calculate signal fraction
    if ithresh == len(mva_threshold_edges)-1: signal_fraction_bins.append(1./len(signal_fraction_edges)/2)
    else: signal_fraction_bins.append(signal_fraction_edges[ithresh-1]+1./len(signal_fraction_edges)/2)
    # Find signal width
    if fixed_width: signal_widths.append([120, 120, 130, 130])
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
    tmva_chain.Draw(f"{branches['observable']}>>hist",f'({branches["y"]}==1&&{mva_window}&&{mass_window})*{branches["weight"]}',"goff")
    #nevents_signal = hist.GetSum()
    nentries_signal = hist.GetEntries()
    bin_s_err = c_double()
    nevents_signal = hist.IntegralAndError(0,hist.GetNbinsX()+1, bin_s_err);
    bin_s_err = bin_s_err.value
    tmva_chain.Draw(f"{branches['observable']}>>hist",f'({branches["y"]}==0&&{mva_window}&&{mass_window})*{branches["weight"]}',"goff")
    bin_b_err = c_double()
    nevents_background = hist.IntegralAndError(0,hist.GetNbinsX()+1, bin_b_err);
    bin_b_err = bin_b_err.value
    #nevents_background = hist.GetSum()
    nentries_background = hist.GetEntries()
    if nevents_background == 0: 
      significances.append(0)
      #significance_errs.append(0)
    else: 
      significances.append(math.sqrt(2*((nevents_signal+nevents_background)*math.log(1+nevents_signal*1./nevents_background)-nevents_signal)))
      significance_errs.append(math.sqrt(1/(significances[-1]**2)*((nevents_signal/nevents_background)**2)*(bin_s_err**2) + 1/(significances[-1]**2)/4*((nevents_signal/nevents_background)**4)*(bin_b_err**2)))
    #print('MVA bin: ',ithresh)
    #print('  ',mva_window)
    #print('  ',mass_window)
    ##print(f'  nSig: {nevents_signal} nBkg: {nevents_background}')
    #print(f'  nSig: {nevents_signal}+-{bin_s_err} nBkg: {nevents_background}+-{bin_b_err}')
    ##print(f'  Entries sig: {nentries_signal} bkg: {nentries_background}')
    #print(f'  Significance: {significances[-1]}+-{significance_errs[-1]}')

  # Find integrated significance
  integrated_significance = 0.
  for significance in significances:
    integrated_significance += significance**2
  integrated_significance = math.sqrt(integrated_significance)
  integrated_significance_err = 0.
  for significance_err in significance_errs:
    integrated_significance_err += significance_err**2
  integrated_significance_err = math.sqrt(integrated_significance_err)
  #print(f'  Integrated significance: {integrated_significance:.4f}+-{integrated_significance_err:.4f}')

  if detail_output: return integrated_significance, integrated_significance_err, [integrated_significance, integrated_significance_err, signal_fraction_bins, significances, significance_errs, signal_widths]
  else: return integrated_significance, integrated_significance_err


# Used branches: {'y', 'yhat', 'weight', 'observable'}
def calculate_mass_shape_difference(root_filename, tree_name, branches, y_value=0, detail_output=False):
  signal_fraction_edges = [0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]

  tmva_chain = ROOT.TChain(tree_name)
  tmva_chain.Add(root_filename)
  mva_threshold_edges = find_signal_fraction_thresholds(signal_fraction_edges, tmva_chain, branches['yhat'], branches['y'], branches['weight'])
  # Add minimum and maximum
  mva_threshold_edges.insert(0,tmva_chain.GetMinimum(branches['yhat']))
  mva_threshold_edges.append(tmva_chain.GetMaximum(branches['yhat']))

  # Storage for detail plots
  mass_histograms = {}
  signal_fraction_bins = []

  for ithresh, mva_threshold in enumerate(mva_threshold_edges):
    if ithresh == 0: continue
    # Make mva window cut
    if ithresh == 1: mva_window = f'{branches["yhat"]}<{mva_threshold}'
    elif ithresh == len(mva_threshold_edges)-1: mva_window = f'{branches["yhat"]}>{mva_threshold_edges[ithresh-1]}'
    else: mva_window = f'{branches["yhat"]}<{mva_threshold}&&{branches["yhat"]}>{mva_threshold_edges[ithresh-1]}'
    # Calculate signal fraction
    if ithresh == len(mva_threshold_edges)-1: signal_fraction_bins.append(1./len(signal_fraction_edges)/2)
    else: signal_fraction_bins.append(signal_fraction_edges[ithresh-1]+1./len(signal_fraction_edges)/2)
    # Hists
    hist = ROOT.TH1F(f"hist_{mva_threshold}_y{y_value}",f"hist_{mva_threshold}_y{y_value}",80,100,180)
    tmva_chain.Draw(f"{branches['observable']}>>hist_{mva_threshold}_y{y_value}",f'{branches["y"]}=={y_value}&&{mva_window}',"goff")
    # Normalize histogram
    sum_weight = hist.GetSumOfWeights()
    if sum_weight != 0:
      hist.Scale(1/sum_weight)
      mass_histograms[mva_threshold] = hist

  # Find std_dev of histograms
  std_values = []
  for iBin in range(next(iter(mass_histograms.values())).GetNbinsX()):
    bin_values = []
    for mva_threshold in mass_histograms:
      hist_ibin_entry = mass_histograms[mva_threshold].GetBinContent(iBin+1)
      bin_values.append(hist_ibin_entry)
      #print("iBin: "+str(iBin+1)+" mva_threshold: "+str(mva_threshold)+" "+str(hist_ibin_entry))
    stdev = np.std(bin_values, dtype=np.float64)
    mean = np.mean(bin_values, dtype=np.float64)
    if (stdev == 0 and mean == 0): continue # Ignore case where hist bins are empty
    std_values.append(stdev)
  mass_shape_difference = np.mean(std_values)
  if detail_output: return mass_shape_difference, [mass_shape_difference, signal_fraction_bins, list(mass_histograms.values())]
  else: return mass_shape_difference

# Used branches: {'y', 'yhat', 'weight', 'observable'}
def calculate_overtraining(root_filename, train_treename, eval_treename, branches, narrow_mass_window=True, detail_output=False):
  nbins = 100
  train_root_file = ROOT.TFile(root_filename)
  train_tree = train_root_file.Get(train_treename)
  eval_root_file = ROOT.TFile(root_filename)
  eval_tree = eval_root_file.Get(eval_treename)

  ROOT.gROOT.cd() # Prevents histograms from disappearing
  cuts = '1'
  if narrow_mass_window: cuts += f"&&{branches['observable']}>122&&{branches['observable']}<128"
  # Find min max of mva variable
  mva_max = max(train_tree.GetMaximum(branches['yhat']), eval_tree.GetMaximum(branches['yhat']))
  mva_min = min(train_tree.GetMinimum(branches['yhat']), eval_tree.GetMinimum(branches['yhat']))

  # Eval tree
  yhat_signal_eval = ROOT.TH1F('signal_eval', 'signal_eval;mva', nbins, mva_min, mva_max)
  yhat_bkg_eval = ROOT.TH1F('bkg_eval', 'bkg_eval', nbins, mva_min, mva_max)
  eval_tree.Draw(f'{branches["yhat"]}>>signal_eval', f"(({branches['y']}==1)&&{cuts})*{branches['weight']}", 'goff')
  eval_tree.Draw(f'{branches["yhat"]}>>bkg_eval', f"(({branches['y']}==0)&&{cuts})*{branches['weight']}", 'goff')
  # Train tree
  yhat_signal_train = ROOT.TH1F('signal_train', 'signal_train', nbins, mva_min, mva_max)
  yhat_bkg_train = ROOT.TH1F('bkg_train', 'bkg_train', nbins, mva_min, mva_max)
  train_tree.Draw(f'{branches["yhat"]}>>signal_train', f"(({branches['y']}==1)&&{cuts})*{branches['weight']}", 'goff')
  train_tree.Draw(f'{branches["yhat"]}>>bkg_train', f"(({branches['y']}==0)&&{cuts})*{branches['weight']}", 'goff')

  # Normalize
  normalize_hist(yhat_signal_eval)
  normalize_hist(yhat_bkg_eval)
  normalize_hist(yhat_signal_train)
  normalize_hist(yhat_bkg_train)

  # Chi2test
  residual_y_signal = array.array('d',[0.]*nbins)
  residual_y_bkg = array.array('d',[0.]*nbins)
  ROOT.gErrorIgnoreLevel = ROOT.kError
  pvalue_signal = yhat_signal_train.Chi2Test(yhat_signal_eval, 'WW', residual_y_signal)
  pvalue_bkg = yhat_bkg_train.Chi2Test(yhat_bkg_eval, 'WW', residual_y_bkg)
  ROOT.gErrorIgnoreLevel = ROOT.kPrint

  detail = [pvalue_signal, pvalue_bkg, nbins, mva_min, mva_max, yhat_signal_train, yhat_bkg_train, yhat_signal_eval, yhat_bkg_eval, residual_y_signal, residual_y_bkg]
  if detail_output: return pvalue_signal, pvalue_bkg, detail
  else: return pvalue_signal, pvalue_bkg

# Used branches: {'y', 'yhat', 'weight', 'observable'}
def calculate_auc_sci(root_filename, tree_name, branches, detail_output=False):
  chain = ROOT.TChain(tree_name)
  chain.Add(root_filename)
  mva_min = chain.GetMinimum(branches['yhat'])
  mva_max = chain.GetMaximum(branches['yhat'])
  bkg_hist = ROOT.TH1D('bkg_hist_aucsci', 'bkg_hist_aucsci', 100, mva_min, mva_max)
  sig_hist = ROOT.TH1D('sig_hist_aucsci', 'sig_hist_aucsci', 100, mva_min, mva_max)
  chain.Draw(f"{branches['yhat']}>>bkg_hist_aucsci",f"({branches['observable']}>122&&{branches['observable']}<128&&{branches['y']}==0)*{branches['weight']}","goff");
  chain.Draw(f"{branches['yhat']}>>sig_hist_aucsci",f"({branches['observable']}>122&&{branches['observable']}<128&&{branches['y']}==1)*{branches['weight']}","goff");
  bkg_hist.Smooth(10)
  sig_hist.Smooth(10)
  if f'-I{os.environ["WORK_DIR"]}/root_scripts' not in ROOT.gSystem.GetIncludePath():
    ROOT.gSystem.AddIncludePath(f'-I{os.environ["WORK_DIR"]}/root_scripts')
  ROOT.gInterpreter.ProcessLine('.L evaluate_mva.C+')
  # results = (auc, sci, bins, areas, signis)
  results = ROOT.get_roc_auc(sig_hist, bkg_hist)
  if detail_output == False: return [results[0], results[1]]
  else: return [results[0], results[1]], results

def plot_feature_shapes(root_filename, tree_name, observable_name, feature_names, weight_name, split_cut, observable_cut, filename_tag):
  nbins = 8
  fraction_edges = [1-(ibin+1)/(nbins) for ibin in range(nbins-1)]

  tmva_chain = ROOT.TChain(tree_name)
  tmva_chain.Add(root_filename)

  # Find min, max for observable. Use 0.05,0.95 quantiles
  x_min_max = []
  hist = ROOT.TH1F(f"hist_{observable_name}",f"hist_{observable_name}",200,0,0)
  tmva_chain.Draw(f'{observable_name}>>hist_{observable_name}','1','goff')
  observable_fractions = [0.01, 0.99]
  observable_quantiles = array.array('d', [0.]*len(observable_fractions))
  observable_fractions = array.array('d', observable_fractions)
  hist.GetQuantiles(len(observable_fractions), observable_quantiles, observable_fractions)
  x_min_max = observable_quantiles.tolist()

  # Find min max of observable. Use if close to 0.05 and 0.95 quantile
  observable_min = tmva_chain.GetMinimum(observable_name)
  observable_max = tmva_chain.GetMaximum(observable_name)
  if (x_min_max[0]-observable_min)/observable_min<0.1: x_min_max[0] = observable_min
  if (observable_max-x_min_max[0])/observable_max<0.1: x_min_max[1] = observable_max

  # Storage for plots
  # display_histograms[feature] = [observable histogram in feature bin]
  display_histograms = {}
  # feature_cut_histograms[feature] = [feature histogram with split cut, feature histogram with observable cut]
  feature_cut_histograms = {}
  # feature_bins[feature] = [feature bin value]
  feature_bins = {}

  # Loop over features
  for feature_name in feature_names:

    # Find feature bins
    feature_threshold_edges = find_sample_fraction_thresholds(fraction_edges, tmva_chain, feature_name, split_cut, weight_name, include_min_max=True)
    feature_bins[feature_name] = feature_threshold_edges

    # Make feature histogram
    feature_cut_histograms[feature_name] = []
    hist_split_cut = ROOT.TH1F(f"hist_{feature_name}_split_cut",f"{feature_name}: {split_cut}",100,feature_bins[feature_name][0],feature_bins[feature_name][-1])
    tmva_chain.Draw(f"{feature_name}>>hist_{feature_name}_split_cut",f'({split_cut})*{weight_name}', 'goff')
    hist_obs_cut = ROOT.TH1F(f"hist_{feature_name}_obs_cut",f"{feature_name}: {observable_cut}",100,feature_bins[feature_name][0],feature_bins[feature_name][-1])
    tmva_chain.Draw(f"{feature_name}>>hist_{feature_name}_obs_cut",f'({observable_cut})*{weight_name}', 'goff')
    normalize_hist(hist_split_cut)
    normalize_hist(hist_obs_cut)
    feature_cut_histograms[feature_name] = [hist_split_cut,hist_obs_cut]

    # Make observable histogram in each feature bin
    for ithresh, feature_threshold in enumerate(feature_threshold_edges):
      if ithresh == 0: continue
      # Make feature window cut
      if ithresh == 1: 
        feature_window = f'{feature_name}<{feature_threshold}'
        feature_window_name = f'val<{feature_threshold:.2f}'
      elif ithresh == len(feature_threshold_edges)-1: 
        feature_window = f'{feature_name}>{feature_threshold_edges[ithresh-1]}'
        feature_window_name = f'val>{feature_threshold_edges[ithresh-1]:.2f}'
      else: 
        feature_window = f'{feature_name}<{feature_threshold}&&{feature_name}>{feature_threshold_edges[ithresh-1]}'
        feature_window_name = f'{feature_threshold_edges[ithresh-1]:.2f}<val<{feature_threshold:.2f}'
      #print(feature_name, feature_window)

      # Make histograms
      hist = ROOT.TH1F(f"hist_{observable_name}_{feature_name}_{feature_window_name}",f"{feature_window_name}",100, x_min_max[0], x_min_max[1])
      tmva_chain.Draw(f"{observable_name}>>hist_{observable_name}_{feature_name}_{feature_window_name}",f'{observable_cut}&&{feature_window}',"goff")
      #print(hist,hist.GetEntries())
      # Normalize histogram
      sum_weight = hist.GetSumOfWeights()
      if sum_weight != 0:
        hist.Scale(1/sum_weight)
        if feature_name not in display_histograms: display_histograms[feature_name] = []
        display_histograms[feature_name].append(hist)

  # Plot histograms
  batch_value = ROOT.gROOT.IsBatch()
  ROOT.gROOT.SetBatch(ROOT.kTRUE)
  for feature_name in display_histograms:
    feature_histograms = display_histograms[feature_name]
    # Make canvas
    c1 = new_canvas()
    c1.SetLeftMargin(0.15)
    # Find min,max of histograms
    hist_min_max = [999,-999]
    for hist in feature_histograms:
      if hist_min_max[0] > hist.GetMinimum(): hist_min_max[0] = hist.GetMinimum()
      if hist_min_max[1] < hist.GetMaximum(): hist_min_max[1] = hist.GetMaximum()
      hist_x_min_max = [hist.GetXaxis().GetXmin(), hist.GetXaxis().GetXmax()]
    # Draw base histogram
    ROOT.gStyle.SetOptStat(0)
    base_hist = ROOT.TH1F(f"base_hist_{observable_name}__{feature_name}",f"{feature_name};{observable_name};Normalized",100,hist_x_min_max[0],hist_x_min_max[1])
    base_hist.SetMinimum(hist_min_max[0])
    base_hist.SetMaximum(hist_min_max[1]*1.1)
    #base_hist.GetXaxis().SetNdivisions(4,4,0,False)
    #base_hist.GetYaxis().SetNdivisions(5,4,0,False)
    base_hist.Draw()
    # Set colors
    colors = [ROOT.kGreen-8, ROOT.kGreen-5, ROOT.kGreen+4,
              ROOT.kBlue-10, ROOT.kBlue-7, 
              ROOT.kRed-10, ROOT.kRed-7, ROOT.kRed,]
    # Draw histogram and legend
    feature_legend = ROOT.TLegend(0.7, 0.6, 0.99, 0.99)
    for iBin, hist in enumerate(feature_histograms):
      hist.SetLineColor(colors[iBin])
      hist.Draw('same')
      feature_legend.AddEntry(hist)
    feature_legend.Draw()
    c1.SaveAs(f"plots/{filename_tag}_{observable_name}_{slugify(feature_name)}.pdf")

    # Plot feature histogram
    c2 = new_canvas()
    c2.SetLeftMargin(0.15)
    # Find min,max of histograms
    hist_min_max = [999,-999]
    for hist in feature_cut_histograms[feature_name]:
      if hist_min_max[0] > hist.GetMinimum(): hist_min_max[0] = hist.GetMinimum()
      if hist_min_max[1] < hist.GetMaximum(): hist_min_max[1] = hist.GetMaximum()
      hist_x_min_max = [hist.GetXaxis().GetXmin(), hist.GetXaxis().GetXmax()]
      #print(feature_name, hist_x_min_max)
    # Draw histogram with split cut and observable cut
    ROOT.gStyle.SetOptStat(0)
    base_hist = ROOT.TH1F(f"base_hist_{feature_name}",f";{feature_name};Normalized",100,hist_x_min_max[0],hist_x_min_max[1])
    base_hist.SetMinimum(hist_min_max[0])
    base_hist.SetMaximum(hist_min_max[1]*1.1)
    base_hist.Draw()
    feature_cut_histograms[feature_name][0].SetLineColor(ROOT.kRed)
    feature_cut_histograms[feature_name][0].Draw("same hist")
    feature_cut_histograms[feature_name][1].Draw("same hist")
    # Draw legend
    feature_legend = ROOT.TLegend(0.7, 0.9, 0.99, 0.99)
    feature_legend.AddEntry(feature_cut_histograms[feature_name][0])
    feature_legend.AddEntry(feature_cut_histograms[feature_name][1])
    feature_legend.Draw()
    # Draw bin cuts
    edge_lines = []
    for iEdge in range(len(feature_bins[feature_name])):
      if iEdge == 0 or iEdge == len(feature_bins[feature_name])-1: continue
      edge_line = ROOT.TLine(feature_bins[feature_name][iEdge],0,feature_bins[feature_name][iEdge],hist_min_max[1])
      edge_lines.append(edge_line)
      edge_line.Draw()
    c2.SaveAs(f"plots/{filename_tag}_{slugify(feature_name)}_bins.pdf")

  ROOT.gROOT.SetBatch(batch_value)


###########################
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
  mva_threshold_edges = find_signal_fraction_thresholds(signal_fraction_edges, tmva_chain, branches['yhat'], branches['y'], branches['weight'])
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
  c1 = new_canvas()
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
  c3 = new_canvas()
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
    c1 = new_canvas()
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
    c3 = new_canvas()
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
    c1 = new_canvas()
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
      c1 = new_canvas()
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
      c1.SaveAs(f"plots/{mva_name}_{name_tag}_hist_mva_{slugify(branch)}.pdf")

def draw_overtraining(overtrain_detail_dict):
  for mva_name in overtrain_detail_dict:
    pvalue_signal, pvalue_bkg, nbins, mva_min, mva_max, yhat_signal_train, yhat_bkg_train, yhat_signal_eval, yhat_bkg_eval,residual_y_signal, residual_y_bkg = overtrain_detail_dict[mva_name]
    y_max = np.amax([yhat_signal_eval.GetMaximum(), yhat_signal_train.GetMaximum(), yhat_bkg_eval.GetMaximum(), yhat_bkg_train.GetMaximum()])
    blank_hist = ROOT.TH1F('blank_hist','', nbins, mva_min, mva_max)
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
  if f'-I{os.environ["WORK_DIR"]}/root_scripts' not in ROOT.gSystem.GetIncludePath():
    ROOT.gSystem.AddIncludePath(f'-I{os.environ["WORK_DIR"]}/root_scripts')
  ROOT.gInterpreter.ProcessLine('.L evaluate_mva.C+')
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
    c1 = new_canvas()
    c1.SetLeftMargin(0.15)
    sig_hist.SetLineColor(ROOT.kRed)
    sig_hist.SetTitle(';MVA value;Normalized')
    normalize_hist(sig_hist)
    normalize_hist(bkg_hist)
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
    c3 = new_canvas()
    c3.SetLeftMargin(0.15)
    colors = [4,800-7,8,1,6,7,8,9,46,49,41,38,30,33]
    for isample, sample_id in enumerate(mva_hists):
      mva_hist = mva_hists[sample_id]
      mva_hist.SetLineColor(colors[isample])
      mva_hist.SetTitle(';MVA value;Normalized')
      normalize_hist(mva_hist)
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
      c2 = new_canvas()
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
      c4 = new_canvas()
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
    c1 = new_canvas()
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
    c1 = new_canvas()
    c1.SetLeftMargin(0.15)
    auc_graph = ROOT.TGraph(len(bins), array.array('d',bins), array.array('d',areas))
    auc_graph.SetTitle("AUC bins;mva value;int_{i}^{end}(signal) #times bkg[i]")
    auc_graph.Draw()
    c1.SaveAs(f"plots/{mva_name}_auc_graph{name_tag}.pdf")
    c2 = new_canvas()
    c2.SetLeftMargin(0.15)
    sci_graph = ROOT.TGraph(len(bins), array.array('d',bins), array.array('d',signis))
    sci_graph.SetTitle("SCI bins;mva value;signal/sqrt(bkg)")
    sci_graph.Draw()
    c2.SaveAs(f"plots/{mva_name}_signis_graph{name_tag}.pdf")
    c3 = new_canvas()
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
    c2 = new_canvas()
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
