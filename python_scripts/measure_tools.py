import ROOT
import array
from ctypes import c_double
import math
import numpy as np

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
  ROOT.gInterpreter.ProcessLine('.L root_scripts/evaluate_mva.C+')
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
