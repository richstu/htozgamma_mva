#!/usr/bin/env python3
import ROOT
from ROOT import TMVA, TFile, TTree, TCut, TChain, TH1F, TString, TCanvas

TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()
import math
import numpy as np
import array
import argparse
import os
import sys
import time
import uproot
import shap
import matplotlib.pyplot as plt

def train_bdt(model_folder, train_filename, test_filename, tmva_filename, weight_name, train_tree_name, eval_tree_name, train_cut, features, bdt_setting):
  output = TFile.Open(tmva_filename, 'RECREATE')
  factory = TMVA.Factory('TMVAClassification', output,
                         '!V:ROC:!Correlations:!Silent:Color:'
                         '!DrawProgressBar:AnalysisType=Classification')
  dataloader = TMVA.DataLoader(model_folder)
  for feature in features:
    dataloader.AddVariable(feature,'F')
  dataloader.SetBackgroundWeightExpression(weight_name)
  dataloader.SetSignalWeightExpression(weight_name)
  dataloader.AddSpectator("lly_m", 'F')
  dataloader.AddSpectator("weightXyear", 'F')
  dataloader.AddSpectator("w_lumiXyear", 'F')
  dataloader.AddSpectator("sampleID", 'I')

  # Add data.
  train_chain = TChain(train_tree_name)
  train_chain.Add(train_filename)
  test_chain = TChain(eval_tree_name)
  test_chain.Add(test_filename)
  dataloader.AddTree(train_chain, 'Background', 1., 'classID==0', TMVA.Types.kTraining)
  dataloader.AddTree(train_chain, 'Signal', 1., 'classID==1', TMVA.Types.kTraining)
  dataloader.AddTree(test_chain, 'Background', 1., 'classID==0', TMVA.Types.kTesting)
  dataloader.AddTree(test_chain, 'Signal', 1., 'classID==1', TMVA.Types.kTesting)
  cut_s = TCut(train_cut);
  cut_b = TCut(train_cut);
  dataloader.PrepareTrainingAndTestTree(cut_s,cut_b,f"NormMode=NumEvents:ScaleWithPreselEff:!V");

  factory.BookMethod(dataloader,TMVA.Types.kBDT,"BDT", bdt_setting);
  factory.TrainAllMethods();
  factory.TestAllMethods();
  factory.EvaluateAllMethods();
  output.Close()

def tmva_shap_importance(root_filename, tree_name, features, tmva_name, tmva_weights, output_tag, nevents=100):
  print("Using SHAP to evaluate feature importance")
  if f'-I{os.environ["WORK_DIR"]}/root_scripts' not in ROOT.gSystem.GetIncludePath():
    ROOT.gSystem.AddIncludePath(f'-I{os.environ["WORK_DIR"]}/root_scripts')
  ROOT.gInterpreter.ProcessLine('.L infer_tmva.C+')
  mva_file = uproot.open(root_filename)
  input_ntuple = mva_file[tree_name].arrays(features)
  input_ntuple = input_ntuple.to_numpy()
  input_ntuple = input_ntuple.view((input_ntuple.dtype[0], len(input_ntuple.dtype.names)))
  infer_tmva = ROOT.infer_tmva_c(tmva_name, tmva_weights)
  def infer_tmva_py_to_c(input_ntuple):
    return infer_tmva.infer(input_ntuple.tolist())
  explainer = shap.Explainer(infer_tmva_py_to_c, input_ntuple[:nevents])
  shap_values = explainer(input_ntuple[:nevents])
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
    bdt_name = 'tmva_bdt_decorr'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva', 'yl_drmin', 'yl_drmax', 'cosThetamass3', 'costheta', 'phi', 'lly_ptmass', 'y_eta', 'l1_eta', 'l2_eta', 'l1_ptmass', 'l2_pt']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    train_cut = '1'
  if method_id == 1: 
    bdt_name = 'tmva_bdt_windowcut_hig19014'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree_baseline'
    eval_tree_name = 'eval_tree_baseline'
    input_mva_ntuple = 'ntuples/ggf_mva_hig19014_decorr_ntuples.root'
    features = ['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'y_ptmass']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    train_cut = '(lly_m>120&&lly_m<130)'
  if method_id == 2: 
    bdt_name = 'tmva_bdt_windowcut_decorr_hig19014'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree_baseline'
    eval_tree_name = 'eval_tree_baseline'
    input_mva_ntuple = 'ntuples/ggf_mva_hig19014_decorr_ntuples.root'
    features = ['y_mva', 'yl_drmin', 'yl_drmax', 'cosThetamass3', 'costheta', 'phi', 'lly_ptmass', 'y_eta', 'l1_eta', 'l2_eta', 'l1_ptmass', 'l2_pt']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    train_cut = '(lly_m>120&&lly_m<130)'
  if method_id == 3: 
    bdt_name = 'tmva_bdt_windowcut_hig19014equal'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree_baseline'
    eval_tree_name = 'eval_tree_baseline'
    input_mva_ntuple = 'ntuples/ggf_mva_hig19014equal_decorr_ntuples.root'
    features = ['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'y_ptmass']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    train_cut = '(lly_m>120&&lly_m<130)'
  print(f'Training with method_id: {method_id}, {bdt_name}')

  start_time = time.time()

  output_model_folder = f'{bdt_name}_model' # Can't be a subfolder
  output_mva_results = f'mva_output/{bdt_name}_factory_results.root'
  print(f'input_mva_ntuple: {input_mva_ntuple}')
  print(f'bdt_model_folder: {output_model_folder}')
  print(f'output_mva_results: {output_mva_results}')

  train_bdt(model_folder=output_model_folder, 
            train_filename=input_mva_ntuple, 
            test_filename=input_mva_ntuple, 
            tmva_filename=output_mva_results,
            weight_name=weight_name,
            train_tree_name=train_tree_name,
            eval_tree_name=eval_tree_name,
            train_cut = train_cut,
            features=features,
            bdt_setting=bdt_setting)
  
  # Run trained bdt over full set
  tree_names = [str(key.split(';')[0]) for key in uproot.open(input_mva_ntuple).keys()]
  if f'-I{os.environ["WORK_DIR"]}/root_scripts' not in ROOT.gSystem.GetIncludePath():
    ROOT.gSystem.AddIncludePath(f'-I{os.environ["WORK_DIR"]}/root_scripts')
  ROOT.gInterpreter.ProcessLine('.L infer_tmva.C+')
  infer_output_filename = f'mva_output/{bdt_name}_results.root'
  ROOT.infer_tmva(f"{input_mva_ntuple}", tree_names, "BDT", f"{output_model_folder}/weights/TMVAClassification_BDT.weights.xml", f"{infer_output_filename}")

  # Use SHAP to evaluate feature importance
  tmva_shap_importance(infer_output_filename, 'train_tree', features, 'BDT', f"{bdt_name}_model/weights/TMVAClassification_BDT.weights.xml", bdt_name)

  elapsed_time = time.time() - start_time
  print(f'Training time: {elapsed_time}')

  #TMVA::TMVAGui("ntuples_mva/TMVA_bdt.root")
