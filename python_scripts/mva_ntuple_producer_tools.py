#!/usr/bin/env python3
import uproot
import numpy as np
import os
import time
import ROOT
import array
import math

import measure_tools

def combine_ntuples(class_filenames):
  # Get input ntuples
  input_ntuples = []
  class_list = []
  sample_list = []
  for class_id, sample_id, filename in class_filenames:
    root_file = uproot.open(filename)
    root_tree = root_file['tree']
    root_branches = root_tree.keys()
    input_ntuples.append(root_tree.arrays(root_branches, '1', library='np'))
    class_list.append(class_id)
    sample_list.append(sample_id)
    root_file.close()

  # Make output ntuple
  output_branches = [branch for branch in root_branches]
  output_ntuple = {}
  for branch in root_branches:
    # Add ntuples into output
    concat_list = []
    for ntuple in input_ntuples:
      concat_list.append(np.squeeze(ntuple[branch]))
    output_ntuple[branch] = np.float32(np.concatenate(concat_list,0))
    # Add classID into output
    concat_list = []
    for iSample, class_id in enumerate(class_list):
      concat_list.append(np.array([class_id]*len(np.squeeze(input_ntuples[iSample][root_branches[0]]))))
    output_ntuple['classID'] = np.int32(np.concatenate(concat_list,0))
    # Add sampleID into output
    concat_list = []
    for iSample, sample_id in enumerate(sample_list):
      concat_list.append(np.array([sample_id]*len(np.squeeze(input_ntuples[iSample][root_branches[0]]))))
    output_ntuple['sampleID'] = np.int32(np.concatenate(concat_list,0))
  output_branches.append('classID')
  output_branches.append('sampleID')

  # Randomize event sequence to have a balance between samples
  random_idx = np.random.permutation(len(output_ntuple['classID']))
  for branch in output_branches:
    output_ntuple[branch] = output_ntuple[branch][random_idx]

  return output_ntuple, output_branches

def split_ntuple(ntuple, branches, baseline_cut, train_cut, weight_branch_names, output_filename):
  # Split ntuple into trees using event number
  train_ntuple, eval_ntuple, test_ntuple = {}, {}, {}
  train_idx = (ntuple['event_number']%314159%3==0) 
  eval_idx =  (ntuple['event_number']%314159%3==1)
  test_idx =  (ntuple['event_number']%314159%3==2)
  for branch in branches:
    train_ntuple[branch] = ntuple[branch][train_idx]
    eval_ntuple[branch] = ntuple[branch][eval_idx]
    test_ntuple[branch] = ntuple[branch][test_idx]
  n_entries = len(ntuple['event_number'])
  train_ntuples = np.stack([np.squeeze(train_ntuple[feat]) for feat in branches], axis=1)
  eval_ntuples = np.stack([np.squeeze(eval_ntuple[feat]) for feat in branches], axis=1)
  test_ntuples = np.stack([np.squeeze(test_ntuple[feat]) for feat in branches], axis=1)

  train_fraction = len(train_ntuples)*1./n_entries
  eval_fraction = len(eval_ntuples)*1./n_entries
  test_fraction = len(test_ntuples)*1./n_entries
  print(f'Total entries: {n_entries} train entries: {len(train_ntuples)}, eval entries: {len(eval_ntuples)}, test entries: {len(test_ntuples)}')
  print(f'  train fraction: {train_fraction:.2f}, val. fraction:{eval_fraction:.2f}, test fraction:{test_fraction:.2f}')

  ## Split the ntuples (Alternative method)
  #train_fraction = 0.34
  #eval_fraction = 0.33
  #test_fraction = 1 - train_fraction - eval_fraction
  #stack_ntuple = np.stack([np.squeeze(ntuple[feat]) for feat in branches], axis=1)
  #n_entries = len(stack_ntuple)
  #train_n_entries = int(train_fraction * n_entries)
  #eval_n_entries = int(eval_fraction * n_entries)
  #test_n_entries = n_entries - train_n_entries - eval_n_entries
  #train_ntuples = stack_ntuple[:train_n_entries]
  #eval_ntuples = stack_ntuple[train_n_entries:train_n_entries+eval_n_entries]
  #test_ntuples = stack_ntuple[train_n_entries+eval_n_entries:]
  #print(f'Total entries: {n_entries} train entries: {len(train_ntuples)}, test entries: {len(test_ntuples)}, eval entries: {len(eval_ntuples)}')

  # Scale up weights
  for weight_branch_name in weight_branch_names:
    weight_branch_index = branches.index(weight_branch_name)
    train_ntuples[:,weight_branch_index] = train_ntuples[:,weight_branch_index] / train_fraction
    test_ntuples[:,weight_branch_index] = test_ntuples[:,weight_branch_index] / test_fraction
    eval_ntuples[:,weight_branch_index] = eval_ntuples[:,weight_branch_index] / eval_fraction

  # Create the ntuples and tmp file
  out_train_ntuples = {}
  out_eval_ntuples = {}
  out_test_ntuples = {}
  for ibranch, branch in enumerate(branches):
    if branch == 'classID' or branch == 'sampleID':
      out_train_ntuples[branch] = np.int32(train_ntuples[:,ibranch])
      out_eval_ntuples[branch] = np.int32(eval_ntuples[:,ibranch])
      out_test_ntuples[branch] = np.int32(test_ntuples[:,ibranch])
    else:
      out_train_ntuples[branch] = np.float32(train_ntuples[:,ibranch])
      out_eval_ntuples[branch] = np.float32(eval_ntuples[:,ibranch])
      out_test_ntuples[branch] = np.float32(test_ntuples[:,ibranch])
  tmp_filename = f'{output_filename}.tmp'
  with uproot.recreate(tmp_filename) as output_file:
    output_file["train_tree_baseline"] = out_train_ntuples
    output_file["test_tree_baseline"] = out_test_ntuples
    output_file["eval_tree_baseline"] = out_eval_ntuples
  print('Temporary saved trees to '+tmp_filename)

  # Apply cut using uproot
  in_file = uproot.open(tmp_filename)

  train_full_tree = in_file['train_tree_baseline']
  train_full_ntuples = train_full_tree.arrays(branches, baseline_cut, library='np')
  train_ntuples = train_full_tree.arrays(branches, train_cut, library='np')

  test_full_tree = in_file['test_tree_baseline']
  test_full_ntuples = test_full_tree.arrays(branches, baseline_cut, library='np')
  test_ntuples = test_full_tree.arrays(branches, train_cut, library='np')

  eval_full_tree = in_file['eval_tree_baseline']
  eval_full_ntuples = eval_full_tree.arrays(branches, baseline_cut, library='np')
  eval_ntuples = eval_full_tree.arrays(branches, train_cut, library='np')

  with uproot.recreate(output_filename) as output_file:
    output_file["train_tree"] = train_ntuples
    output_file["train_tree_baseline"] = train_full_ntuples
    output_file["test_tree"] = test_ntuples
    output_file["test_tree_baseline"] = test_full_ntuples
    output_file["eval_tree"] = eval_ntuples
    output_file["eval_tree_baseline"] = eval_full_ntuples
  print('Tree saved to '+output_filename)
  in_file.close()

def split_ntuple_train_eval(ntuple, branches, baseline_cut, train_cut, weight_branch_names, output_filename):
  # Split ntuple into trees using event number
  train_ntuple, eval_ntuple, test_ntuple = {}, {}, {}
  train_idx = (ntuple['event_number']%314159%2==0) 
  eval_idx =  (ntuple['event_number']%314159%2==1)
  for branch in branches:
    train_ntuple[branch] = ntuple[branch][train_idx]
    eval_ntuple[branch] = ntuple[branch][eval_idx]
  n_entries = len(ntuple['event_number'])
  train_ntuples = np.stack([np.squeeze(train_ntuple[feat]) for feat in branches], axis=1)
  eval_ntuples = np.stack([np.squeeze(eval_ntuple[feat]) for feat in branches], axis=1)

  train_fraction = len(train_ntuples)*1./n_entries
  eval_fraction = len(eval_ntuples)*1./n_entries
  print(f'Total entries: {n_entries} train entries: {len(train_ntuples)}, eval entries: {len(eval_ntuples)}')
  print(f'  train fraction: {train_fraction:.2f}, val. fraction:{eval_fraction:.2f}')

  ## Split the ntuples (Alternative method)
  #train_fraction = 0.34
  #eval_fraction = 0.33
  #test_fraction = 1 - train_fraction - eval_fraction
  #stack_ntuple = np.stack([np.squeeze(ntuple[feat]) for feat in branches], axis=1)
  #n_entries = len(stack_ntuple)
  #train_n_entries = int(train_fraction * n_entries)
  #eval_n_entries = int(eval_fraction * n_entries)
  #test_n_entries = n_entries - train_n_entries - eval_n_entries
  #train_ntuples = stack_ntuple[:train_n_entries]
  #eval_ntuples = stack_ntuple[train_n_entries:train_n_entries+eval_n_entries]
  #test_ntuples = stack_ntuple[train_n_entries+eval_n_entries:]
  #print(f'Total entries: {n_entries} train entries: {len(train_ntuples)}, test entries: {len(test_ntuples)}, eval entries: {len(eval_ntuples)}')

  # Scale up weights
  for weight_branch_name in weight_branch_names:
    weight_branch_index = branches.index(weight_branch_name)
    train_ntuples[:,weight_branch_index] = train_ntuples[:,weight_branch_index] / train_fraction
    eval_ntuples[:,weight_branch_index] = eval_ntuples[:,weight_branch_index] / eval_fraction

  # Create the ntuples and tmp file
  out_train_ntuples = {}
  out_eval_ntuples = {}
  for ibranch, branch in enumerate(branches):
    if branch == 'classID':
      out_train_ntuples[branch] = np.int32(train_ntuples[:,ibranch])
      out_eval_ntuples[branch] = np.int32(eval_ntuples[:,ibranch])
    else:
      out_train_ntuples[branch] = np.float32(train_ntuples[:,ibranch])
      out_eval_ntuples[branch] = np.float32(eval_ntuples[:,ibranch])
  tmp_filename = f'{output_filename}.tmp'
  with uproot.recreate(tmp_filename) as output_file:
    output_file["train_tree_baseline"] = out_train_ntuples
    output_file["eval_tree_baseline"] = out_eval_ntuples
  print('Temporary saved trees to '+tmp_filename)

  # Apply cut using uproot
  in_file = uproot.open(tmp_filename)

  train_full_tree = in_file['train_tree_baseline']
  train_full_ntuples = train_full_tree.arrays(branches, baseline_cut, library='np')
  train_ntuples = train_full_tree.arrays(branches, train_cut, library='np')

  eval_full_tree = in_file['eval_tree_baseline']
  eval_full_ntuples = eval_full_tree.arrays(branches, baseline_cut, library='np')
  eval_ntuples = eval_full_tree.arrays(branches, train_cut, library='np')

  #print(f'train_full: {train_full_ntuples}')
  #print(f'train: {train_ntuples}')
  with uproot.recreate(output_filename) as output_file:
    output_file["train_tree"] = train_ntuples
    output_file["train_tree_baseline"] = train_full_ntuples
    output_file["eval_tree"] = eval_ntuples
    output_file["eval_tree_baseline"] = eval_full_ntuples
  print('Tree saved to '+output_filename)
  in_file.close()

def make_alt_mva_ntuples(ntuple_filename, ntuple_additional_filename, reweight_sample_ids, reweight_scale, weight_branch_names, output_filename):
  # Combine two files
  combine_tree_names = ['train_tree', 'train_tree_baseline', 'eval_tree', 'eval_tree_baseline']
  other_tree_names = ['test_tree', 'test_tree_baseline']

  # Collect ntuples
  print("Input")
  # tree_ntuples[tree] = [{branch: [[]]}]
  tree_ntuples = {}
  for tree_name in combine_tree_names:
    ntuple_file = uproot.open(ntuple_filename)
    ntuple_tree = ntuple_file[tree_name]
    ntuple_branches = ntuple_tree.keys()
    tree_ntuples[tree_name] = [ntuple_tree.arrays(ntuple_branches, '1', library='np')]
    ntuple_additional_file = uproot.open(ntuple_additional_filename)
    ntuple_additional_tree = ntuple_additional_file[tree_name]
    tree_ntuples[tree_name].append(ntuple_additional_tree.arrays(ntuple_branches, '1', library='np'))
    print(f'  {ntuple_filename}:{tree_name} nevts: {ntuple_tree.num_entries}')
    print(f'  {ntuple_additional_filename}:{tree_name} nevts: {ntuple_additional_tree.num_entries}')
    ntuple_file.close()
  for tree_name in other_tree_names:
    ntuple_file = uproot.open(ntuple_filename)
    ntuple_tree = ntuple_file[tree_name]
    ntuple_branches = ntuple_tree.keys()
    tree_ntuples[tree_name] = [ntuple_tree.arrays(ntuple_branches, '1', library='np')]
    print(f'  {ntuple_filename}:{tree_name} nevts: {ntuple_tree.num_entries}')
    ntuple_file.close()

  # Combine ntuple
  combine_branches = [branch for branch in ntuple_branches]
  # combine_ntuple[tree][branch] = []
  combine_ntuple = {}
  for tree_name in combine_tree_names:
    combine_ntuple[f'{tree_name}_with_alt'] = {}
    combine_ntuple[tree_name] = {}
    for branch in combine_branches:
      # concat_list = [[value], ...]
      concat_list = []
      for ntuple in tree_ntuples[tree_name]:
        concat_list.append(np.squeeze(ntuple[branch]))
      if branch == 'classID' or branch == 'sampleID':
        combine_ntuple[f'{tree_name}_with_alt'][branch] = np.int32(np.concatenate(concat_list,0))
        combine_ntuple[f'{tree_name}'][branch] = np.int32(concat_list[0])
      else:
        combine_ntuple[f'{tree_name}_with_alt'][branch] = np.float32(np.concatenate(concat_list,0))
        combine_ntuple[f'{tree_name}'][branch] = np.float32(concat_list[0])
  for tree_name in other_tree_names:
    combine_ntuple[tree_name] = {}
    for branch in combine_branches:
      if branch == 'classID' or branch == 'sampleID':
        combine_ntuple[f'{tree_name}'][branch] = np.int32(np.squeeze(tree_ntuples[tree_name][0][branch]))
      else:
        combine_ntuple[f'{tree_name}'][branch] = np.float32(np.squeeze(tree_ntuples[tree_name][0][branch]))

  # Randomize event sequence to have a balance between samples
  for tree_name in combine_tree_names:
    randomize_tree_name = f'{tree_name}_with_alt'
    random_idx = np.random.permutation(len(combine_ntuple[randomize_tree_name]['classID']))
    for branch in combine_branches:
      combine_ntuple[randomize_tree_name][branch] = combine_ntuple[randomize_tree_name][branch][random_idx]

  # stack_ntuples[tree_name] = [[value, value, ...] for event, ...]
  stack_ntuple = {}
  for tree_name in combine_tree_names:
    stack_tree_name = f'{tree_name}_with_alt'
    stack_ntuple[stack_tree_name] =  np.stack([np.squeeze(combine_ntuple[stack_tree_name][feat]) for feat in combine_branches], axis=1)

  # Weight down alt by half for sampleID
  for tree_name in combine_tree_names:
    wgtdown_tree_name = f'{tree_name}_with_alt'
    for weight_branch_name in weight_branch_names:
      sample_branch_index = combine_branches.index('sampleID')
      reweight_mask = np.logical_or.reduce([stack_ntuple[wgtdown_tree_name][:,sample_branch_index] == i for i in reweight_sample_ids])
      reweight = np.where(reweight_mask, reweight_scale, 1)
      weight_branch_index = combine_branches.index(weight_branch_name)
      stack_ntuple[wgtdown_tree_name][:,weight_branch_index] = stack_ntuple[wgtdown_tree_name][:,weight_branch_index] * reweight

  # Create ntuples
  #output_ntuple[tree][branch] = [value]
  output_ntuple = {}
  # unstack alt ntuples
  for tree_name in combine_tree_names:
    unstack_tree_name = f'{tree_name}_with_alt'
    output_ntuple[unstack_tree_name] = {}
    for ibranch, branch in enumerate(combine_branches):
      if branch == 'classID' or branch == 'sampleID':
        output_ntuple[unstack_tree_name][branch] = np.int32(stack_ntuple[unstack_tree_name][:,ibranch])
      else:
        output_ntuple[unstack_tree_name][branch] = np.float32(stack_ntuple[unstack_tree_name][:,ibranch])
  # Add in other ntuples
  for tree_name in combine_tree_names: output_ntuple[tree_name] = combine_ntuple[tree_name]
  for tree_name in other_tree_names: output_ntuple[tree_name] = combine_ntuple[tree_name]

  with uproot.recreate(output_filename) as output_file:
    for tree_name in output_ntuple:
      output_file[tree_name] = output_ntuple[tree_name]
  print('Tree saved to '+output_filename)

  for tree_name in output_ntuple: 
    print('  ',tree_name, len(output_ntuple[tree_name][combine_branches[0]]))

# Makes weights that change base distribution to target distribution
def get_same_shape_weights(target_filename, target_tree, target_branch, target_weight, target_cut,
                           base_filename, base_tree, base_branch, base_weight, base_cut,
                           output_weight_filename):
  nbins = 160
  # Open files
  target_chain = ROOT.TChain(target_tree)
  target_chain.Add(target_filename)
  base_chain = ROOT.TChain(base_tree)
  base_chain.Add(base_filename)
  # Find min max
  min_value = min(target_chain.GetMinimum(target_branch), base_chain.GetMinimum(base_branch))
  max_value = max(target_chain.GetMaximum(target_branch), base_chain.GetMaximum(base_branch))

  # Make histograms
  base_hist = ROOT.TH1F('base_hist', 'base_hist', nbins, min_value, max_value)
  base_chain.Draw(f'{base_branch} >> base_hist', f'({base_cut})*{base_weight}', 'goff')
  target_hist = ROOT.TH1F('target_hist', 'target_hist', nbins, min_value, max_value)
  target_chain.Draw(f'{target_branch} >> target_hist', f'({target_cut})*{target_weight}', 'goff')

  # Normalize hists
  measure_tools.normalize_hist(base_hist)
  measure_tools.normalize_hist(target_hist)

  # Make weights
  weight_hist = target_hist.Clone('weight_hist')
  weight_hist.Divide(base_hist)

  # Limit weight to [20, -20]
  nbins = weight_hist.GetNbinsX()
  for ibin in range(nbins+2):
    value = weight_hist.GetBinContent(ibin)
    if value > 20: weight_hist.SetBinContent(ibin,20)
    if value < -20: weight_hist.SetBinContent(ibin,-20)

  # Save weights to file
  weight_file = ROOT.TFile(output_weight_filename, 'recreate')
  weight_hist.Write()
  base_hist.Write()
  target_hist.Write()
  weight_file.Close()
  print(f'Created {output_weight_filename}')


def init_same_shape_weights_functions():
  ROOT.gInterpreter.Declare("""
  template <class C>
  using RVec = ROOT::VecOps::RVec<C>;
  
  TFile * weight_file = 0;
  TH1F * weight_hist = 0;
  int apply_class_index = -1;
  
  void setup_shape_weight(string weight_filename) {
    weight_file = new TFile(weight_filename.c_str());
    weight_hist = (TH1F*)weight_file->Get("weight_hist");
  }

  void setup_apply_class_index(int in_apply_class_index) {
    apply_class_index = in_apply_class_index;
  }
  
  float get_shape_weight(const Float_t & lly_m, const Int_t & classID) {
    if (classID != apply_class_index) return 1.;
    //std::cout<<lly_m<<std::endl;
    //std::cout<<weight_hist->GetBinContent(weight_hist->FindBin(120))<<std::endl;
    float shape_weight = weight_hist->GetBinContent(weight_hist->FindBin(lly_m));
    return shape_weight;
  }
  """)

def apply_same_shape_weights(weight_filename, input_filename, nominal_weight_name, shape_weight_name, w_shape_name, apply_class_index, output_filename):
  ROOT.setup_shape_weight(weight_filename)
  ROOT.setup_apply_class_index(apply_class_index)
  defines = [(w_shape_name,'get_shape_weight(lly_m, classID)'),
             (shape_weight_name, f'{nominal_weight_name}*w_same_shape')]
  tree_names = [str(key.split(';')[0]) for key in uproot.open(input_filename).keys()]
  for itree, tree_name in enumerate(tree_names):
    df = ROOT.RDataFrame(tree_name, input_filename)
    for define in defines:
      df = df.Define(define[0],define[1])
    if itree == 0:
      df.Snapshot(tree_name,output_filename, df.GetColumnNames())
    else:
      snapshot_options = ROOT.RDF.RSnapshotOptions()
      snapshot_options.fMode = "update";
      df.Snapshot(tree_name,output_filename, df.GetColumnNames(), snapshot_options)
  print('Wrote '+output_filename)

def get_decorr_func(target_filename, target_tree, target_variable, target_decorr_variables, 
                    target_weight, target_cut,
                    output_relation_filename):
  nbins = 100
  # Open files
  target_chain = ROOT.TChain(target_tree)
  target_chain.Add(target_filename)
  # Find min max
  min_value = target_chain.GetMinimum(target_variable)
  max_value = target_chain.GetMaximum(target_variable)

  tgraph_dict = {}
  fitresult_dict = {}

  var_nbins = 10
  sample_cut = 'classID==0'
  weight_name = 'weightXyear'
  for decorr_variable in target_decorr_variables:
    if ':' in decorr_variable:
      decorr_variable = decorr_variable.split(':')[0]
    # Find quantiles for decorr_variable with var_nbins
    sample_fraction_edges = [1-(ibin+1)/(var_nbins) for ibin in range(var_nbins-1)]
    decorr_var_threshold_edges = measure_tools.find_sample_fraction_thresholds(sample_fraction_edges, target_chain, decorr_variable, sample_cut, weight_name, include_min_max=True)

    decorr_variable_name = measure_tools.slugify(decorr_variable)

    # Find average target_variable in each decorr_variable
    # relation_points = [(decorr_var_value, target_var_value)]
    relation_points = []
    for ithresh, decorr_var_threshold in enumerate(decorr_var_threshold_edges):
      if ithresh == 0: continue
      # Make decorr_variable window cut
      if ithresh == 1: 
        decorr_var_window = f'{decorr_variable}<{decorr_var_threshold}'
        decorr_var_val = (decorr_var_threshold+decorr_var_threshold_edges[0])/2
      elif ithresh == len(decorr_var_threshold_edges)-1: 
        decorr_var_window = f'{decorr_variable}>{decorr_var_threshold_edges[ithresh-1]}'
        decorr_var_val = (decorr_var_threshold_edges[ithresh-1]+decorr_var_threshold_edges[-1])/2
      else: 
        decorr_var_window = f'{decorr_variable}<{decorr_var_threshold}&&{decorr_variable}>{decorr_var_threshold_edges[ithresh-1]}'
        decorr_var_val = (decorr_var_threshold+decorr_var_threshold_edges[ithresh-1])/2
      # Should ignore first and last bin because center value can be off.
      if ithresh == 1 : continue
      if ithresh == len(decorr_var_threshold_edges)-1: continue
      hist = ROOT.TH1F(f"hist_{target_variable}_{decorr_variable_name}",f"hist_{target_variable}_{decorr_variable}",nbins,min_value,max_value)
      target_chain.Draw(f'{target_variable}>>hist_{target_variable}_{decorr_variable_name}',f'({sample_cut}&&{decorr_var_window})*{weight_name}',"goff")
      # Get median
      target_var_quantiles = array.array('d', [0.])
      hist.GetQuantiles(1, target_var_quantiles, array.array('d', [0.5])) # Use lower value to expand lly_m range
      relation_points.append([target_var_quantiles[0],decorr_var_val])

    # Make TGraph using relation points, x=mass, y=decorr_val
    x_array = array.array('d',[x_val for x_val, y_val in relation_points])
    y_array = array.array('d',[y_val for x_val, y_val in relation_points])
    relation_graph = ROOT.TGraph(len(x_array), x_array, y_array)
    relation_graph.SetName(f'graph_{decorr_variable_name}')
    # Fit points with 3rd-order polymonial var = f(mass)
    fit_result = relation_graph.Fit("1++x++x*x++x*x*x","S")
    fit_result.SetName(f'func_{decorr_variable_name}')

    tgraph_dict[decorr_variable_name] = relation_graph
    fitresult_dict[decorr_variable_name] = fit_result
  
  # Save parameters of 4th-order polymonial var = f(mass)
  # Save TGraphs of relation points
  relation_file = ROOT.TFile(output_relation_filename, 'recreate')
  for variable in tgraph_dict:
    tgraph_dict[variable].Write()
    fitresult_dict[variable].Write()
  relation_file.Close()
  print(f'Saved to {output_relation_filename}')

def init_apply_decorr_func():
  ROOT.gInterpreter.Declare("""
  TFile * m_decorr_func_file = 0; 
  vector<TF1 *> m_decorr_funcs;
  vector<float> m_mass_shift;

  void setup_decorr_func(vector<string> decorr_variables, string decorr_func_filename) {
    m_decorr_funcs.clear();
    m_decorr_func_file = new TFile(decorr_func_filename.c_str());
    for (string decorr_variable : decorr_variables) {
      m_decorr_funcs.push_back(((TF1*)((TGraph*) m_decorr_func_file->Get(("graph_"+decorr_variable).c_str()))->GetListOfFunctions()->At(0)));
      cout<<"Loaded graph_"<<decorr_variable<<endl;
    }
    // Shift mass if func value is below 0 at mass=100
    m_mass_shift = vector<float>(m_decorr_funcs.size());
    for (unsigned ivar = 0; ivar < m_mass_shift.size(); ++ivar) {
      // Find where func becomes 0.
      for (int mass = 100; mass < 130; ++mass) {
        if (m_decorr_funcs[ivar]->Eval(mass)>0) {
          cout<<"ivar: "<<ivar<<" mass: "<<mass<<" decorr_func_value:"<<m_decorr_funcs[ivar]->Eval(mass)<<" shift mass by:"<<mass-100<<endl;
          m_mass_shift[ivar] = mass-100;
          break;
        }
      }
    }
  }

  float get_decorr_value(const Int_t & decorr_var_index, const Float_t & decorr_var, const Float_t & lly_m) {
    float eval_value = m_decorr_funcs[decorr_var_index]->Eval(lly_m+m_mass_shift[decorr_var_index]);
    //cout<<m_decorr_funcs.size()<<" "<<decorr_var_index<<" "<<decorr_var<<" "<<lly_m<<" "<<m_decorr_funcs[decorr_var_index]->Eval(lly_m+m_mass_shift[decorr_var_index])<<" "<<decorr_var/m_decorr_funcs[decorr_var_index]->Eval(lly_m+m_mass_shift[decorr_var_index])<<endl;
    return decorr_var/eval_value;
  }
  """)

def apply_decorr_func(decorr_variables, decorr_func_filename, decorr_defines, decorr_cuts, input_filename, output_filename):
  derive_decorr_variables = []
  for decorr_variable in decorr_variables:
    if ':' in decorr_variable: derive_decorr_variable = decorr_variable.split(':')[0]
    else: derive_decorr_variable = decorr_variable
    derive_decorr_variables.append(measure_tools.slugify(derive_decorr_variable))
  ROOT.setup_decorr_func(derive_decorr_variables, decorr_func_filename)
  defines = decorr_defines.copy()
  # Make var' = var/f(mass)
  for ivar, decorr_variable in enumerate(decorr_variables):
    if ':' in decorr_variable: apply_decorr_variable = decorr_variable.split(':')[1]
    else: apply_decorr_variable = decorr_variable
    defines.append((f'{measure_tools.slugify(apply_decorr_variable)}_decorr', f'get_decorr_value({ivar}, {apply_decorr_variable},lly_m)'))
  tree_names = [str(key.split(';')[0]) for key in uproot.open(input_filename).keys()]
  for itree, tree_name in enumerate(tree_names):
    df = ROOT.RDataFrame(tree_name, input_filename)
    for define in defines:
      df = df.Define(define[0],define[1])
    for cut in decorr_cuts:
      df = df.Filter(cut)
    if itree == 0:
      df.Snapshot(tree_name,output_filename, df.GetColumnNames())
    else:
      snapshot_options = ROOT.RDF.RSnapshotOptions()
      snapshot_options.fMode = "update";
      df.Snapshot(tree_name,output_filename, df.GetColumnNames(), snapshot_options)
  print('Wrote '+output_filename)

