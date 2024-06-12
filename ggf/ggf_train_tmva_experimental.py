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

def train_bdt(model_folder, train_filename, test_filename, tmva_filename, weight_name, train_tree_name, eval_tree_name, train_cut, features, spectators, prepare_tree_setting, bdt_setting):
  output = TFile.Open(tmva_filename, 'RECREATE')
  factory = TMVA.Factory('TMVAClassification', output,
                         '!V:ROC:!Correlations:!Silent:Color:'
                         '!DrawProgressBar:AnalysisType=Classification')
  dataloader = TMVA.DataLoader(model_folder)
  for feature in features:
    dataloader.AddVariable(feature,'F')
  dataloader.SetBackgroundWeightExpression(weight_name)
  dataloader.SetSignalWeightExpression(weight_name)
  for spectator in spectators:
    dataloader.AddSpectator(spectator, 'F')
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
  #dataloader.PrepareTrainingAndTestTree(cut_s,cut_b,f"NormMode=NumEvents:ScaleWithPreselEff:!V");
  dataloader.PrepareTrainingAndTestTree(cut_s,cut_b,f"NormMode=None:ScaleWithPreselEff:!V");

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
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_eta', 'l1_eta', 'l2_eta']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 1: 
    bdt_name = 'tmva_bdt_windowcut_hig19014'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree_baseline'
    eval_tree_name = 'eval_tree_baseline'
    input_mva_ntuple = 'ntuples/ggf_mva_hig19014_decorr_ntuples.root'
    features = ['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'y_ptmass']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '(lly_m>120&&lly_m<130)'
  if method_id == 2: 
    bdt_name = 'tmva_bdt_windowcut_decorr_hig19014'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree_baseline'
    eval_tree_name = 'eval_tree_baseline'
    input_mva_ntuple = 'ntuples/ggf_mva_hig19014_decorr_ntuples.root'
    features = ['y_mva', 'yl_drmin', 'yl_drmax', 'cosThetamass3', 'costheta', 'phi', 'lly_ptmass', 'y_eta', 'l1_eta', 'l2_eta', 'l2_pt']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '(lly_m>120&&lly_m<130)'
  if method_id == 3: 
    bdt_name = 'tmva_bdt_nowindowcut'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree_baseline'
    eval_tree_name = 'eval_tree_baseline'
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 4: 
    bdt_name = 'tmva_bdt_windowcut'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 5: 
    bdt_name = 'tmva_bdt_yptmass3'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_mva_yptmass3_decorr_ntuples.root'
    features = ['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 6: 
    bdt_name = 'tmva_bdt_yptmass2'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_mva_yptmass2_decorr_ntuples.root'
    features = ['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 7: 
    bdt_name = 'tmva_bdt_decorr_cosThetacut'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_mva_cosTheta_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosTheta', 'costheta', 'phi', 'lly_ptmass', 'y_eta', 'l1_eta', 'l2_eta']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 8: 
    bdt_name = 'tmva_bdt_decorr19'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'l1_ptmass', 'l2_pt', 'leplep_flavor', 'leplep_m', 'leplep_eta', 'lly_ptt', 'lly_eta', 'ht']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 9: 
    bdt_name = 'tmva_bdt_decorr_min19'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'ht']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 10: 
    bdt_name = 'tmva_bdt_decorr_min19bad'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'lly_ptt', 'ht']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 11: 
    bdt_name = 'tmva_bdt_min14_ypt'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'ht', 'y_ptinfo2mass2']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 12: 
    bdt_name = 'tmva_bdt_min13'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 13: 
    bdt_name = 'tmva_bdt_decorr18'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'l1_ptmass', 'l2_pt', 'leplep_flavor', 'leplep_m', 'leplep_eta', 'lly_ptt', 'lly_eta']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 14: 
    bdt_name = 'tmva_bdt_min13_ypt'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt','y_ptinfo2mass2']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 15: 
    bdt_name = 'tmva_bdt19'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'l1_ptmass', 'l2_pt', 'leplep_flavor', 'leplep_m', 'leplep_eta', 'lly_ptt', 'lly_eta','y_ptinfo2mass2']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 16: 
    bdt_name = 'tmva_bdtg_min13'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=850:MinNodeSize=4%:MaxDepth=4:BoostType=Grad:Shrinkage=0.1:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=20"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 17: 
    bdt_name = 'tmva_bdtg_min13_ypt'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt','y_ptinfo2mass2']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=850:MinNodeSize=4%:MaxDepth=4:BoostType=Grad:Shrinkage=0.1:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=20"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 18: 
    bdt_name = 'tmva_bdtg_yptmass'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'y_ptmass']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=850:MinNodeSize=4%:MaxDepth=4:BoostType=Grad:Shrinkage=0.1:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=20"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 19: 
    bdt_name = 'tmva_bdt_tight_higwindow_var11'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 20: 
    bdt_name = 'tmva_bdta_tight_higwindow_var12'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'y_ptmass']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 21: 
    bdt_name = 'tmva_bdtg_tight_higwindow_var12'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'y_ptmass']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=850:MinNodeSize=4%:MaxDepth=4:BoostType=Grad:Shrinkage=0.1:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=20"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 22: 
    bdt_name = 'tmva_bdtg_tight_higwindow_var11'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=850:MinNodeSize=4%:MaxDepth=4:BoostType=Grad:Shrinkage=0.1:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=20"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 23: 
    bdt_name = 'tmva_bdta_tight_higwindow_var12cosh'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'y_ptinfo2mass2']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 24: 
    bdt_name = 'tmva_bdtg_tight_higwindow_var12cosh'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','phi','y_res','y_eta','l1_eta','l2_eta', 'y_ptinfo2mass2']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=850:MinNodeSize=4%:MaxDepth=4:BoostType=Grad:Shrinkage=0.1:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=20"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 25: 
    bdt_name = 'tmva_bdta_tight_higwindow_var13'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 26: 
    bdt_name = 'tmva_bdtg_tight_higwindow_var13'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=850:MinNodeSize=4%:MaxDepth=4:BoostType=Grad:Shrinkage=0.1:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=20"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 27: 
    bdt_name = 'tmva_bdta_tight0j_higwindow_var13'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_tight0j_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 28: 
    bdt_name = 'tmva_bdta_tight1j_higwindow_var13'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_tight1j_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 29: 
    bdt_name = 'tmva_bdta_tight_higwindow_var12'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear', 'costheta']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 30: 
    bdt_name = 'tmva_bdta_tight_higwindow_var10'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva', 'cosThetamass', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear', 'costheta', 'yl_drmin', 'yl_drmax']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 31: 
    bdt_name = 'tmva_bdta_tight_higwindow_var14'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'npv']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 32: 
    bdt_name = 'tmva_bdta_tight_higwindow_var7'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva', 'cosThetamass', 'phi', 'lly_ptmass', 'y_res', 'leplep_m', 'lly_ptt']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear', 'costheta', 'yl_drmin', 'yl_drmax', 'y_eta', 'l1_eta', 'l2_eta']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 33: 
    bdt_name = 'tmva_bdta_tight_higwindow_var6'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva', 'cosThetamass', 'phi', 'lly_ptmass', 'leplep_m', 'lly_ptt']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear', 'costheta', 'yl_drmin', 'yl_drmax', 'y_eta', 'l1_eta', 'l2_eta', 'y_res']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 34: 
    bdt_name = 'tmva_bdta_tight_higwindow_var5'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva', 'cosThetamass', 'phi', 'lly_ptmass', 'leplep_m']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear', 'costheta', 'yl_drmin', 'yl_drmax', 'y_eta', 'l1_eta', 'l2_eta', 'y_res', 'lly_ptt']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 35: 
    bdt_name = 'tmva_bdta_tight_higwindow_var3'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_mva_decorr_ntuples.root'
    features = ['y_mva', 'phi', 'lly_ptmass']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear', 'costheta', 'yl_drmin', 'yl_drmax', 'y_eta', 'l1_eta', 'l2_eta', 'y_res', 'lly_ptt', 'cosThetamass', 'leplep_m']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 36: 
    bdt_name = 'tmva_bdta_tight1j_higwindow_var19'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggf_tight1j_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'j1_pt', 'j1_eta', 'j1_phi', 'llyj_dphi', 'yj1_deta', 'yj1_dr', 'llyj1_ptbal']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 37: 
    bdt_name = 'tmva_bdta_ggfvbf_tight1j_higwindow_var19'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggfvbf_tight1j_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'j1_pt', 'j1_eta', 'j1_phi', 'llyj_dphi', 'yj1_deta', 'yj1_dr', 'llyj1_ptbal']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 38: 
    bdt_name = 'tmva_bdta_ggfvbf_tight0j_higwindow_var13'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggfvbf_tight0j_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 39: 
    bdt_name = 'tmva_bdta_ggfvbf_tight1j_higwindow_var13'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggfvbf_tight1j_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 40: 
    bdt_name = 'tmva_bdta_ggfvbf_tight_higwindow_var13'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggfvbf_tight_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 41: 
    bdt_name = 'tmva_bdta_ggfvbf_tight1j_higwindow_var15'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggfvbf_tight1j_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 42: 
    bdt_name = 'tmva_bdta_ggfvbf_tight_higwindow_var15'
    weight_name = 'weightXyear'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggfvbf_tight_decorr_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 43: 
    bdt_name = 'tmva_bdta_wgt11_ggfvbf_tight_higwindow_var15'
    weight_name = 'wgt11'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggfvbf_tight_shape_decorr_weight_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 44: 
    bdt_name = 'tmva_bdta_wgt_ggfvbf_tight_higwindow_var15'
    weight_name = 'wgt'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggfvbf_tight_shape_decorr_weight_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 45: 
    bdt_name = 'tmva_bdta_wgteq1_ggfvbf_tight_higwindow_var15'
    weight_name = 'wgteq1'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggfvbf_tight_shape_decorr_weight_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
  if method_id == 46: 
    bdt_name = 'tmva_bdta_wgtsigres11_ggfvbf_tight_higwindow_var15'
    weight_name = 'wgtsigres11'
    train_tree_name = 'train_tree'
    eval_tree_name = 'eval_tree'
    input_mva_ntuple = 'ntuples/ggfvbf_tight_shape_decorr_weight_ntuples.root'
    features = ['y_mva', 'yl_drminmass', 'yl_drmax', 'cosThetamass', 'costheta', 'phi', 'lly_ptmass', 'y_res', 'y_eta', 'l1_eta', 'l2_eta', 'leplep_m', 'lly_ptt', 'llyj_dphi', 'yj1_dr']
    spectators = ['lly_m', 'weightXyear', 'w_lumiXyear']
    bdt_setting = "!H:!V:NTrees=350:MinNodeSize=4%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning"
    prepare_tree_setting = "NormMode=NumEvents:ScaleWithPreselEff:!V"
    train_cut = '1'
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
            spectators=spectators,
            prepare_tree_setting=prepare_tree_setting,
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
