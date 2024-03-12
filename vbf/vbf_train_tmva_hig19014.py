#!/usr/bin/env python3
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

def train_bdt(model_folder, train_filename, test_filename, tmva_filename):
  output = TFile.Open(tmva_filename, 'RECREATE')
  factory = TMVA.Factory('TMVAClassification', output,
                         '!V:ROC:!Correlations:!Silent:Color:'
                         '!DrawProgressBar:AnalysisType=Classification')
  dataloader = TMVA.DataLoader(model_folder)
  dataloader.AddVariable("y_mva",'F')
  dataloader.AddVariable("yl_drmin",'F')
  dataloader.AddVariable("yl_drmax",'F')
  dataloader.AddVariable("cosTheta",'F')
  dataloader.AddVariable("costheta",'F')
  dataloader.AddVariable("phi",'F')
  dataloader.AddVariable("y_res",'F')
  dataloader.AddVariable("y_eta",'F')
  dataloader.AddVariable("l1_eta",'F')
  dataloader.AddVariable("l2_eta",'F')
  dataloader.AddVariable("lly_ptt",'F')
  dataloader.AddVariable("jj_deta",'F')
  dataloader.AddVariable("jj_dphi",'F')
  dataloader.AddVariable("yj1_dr",'F')
  dataloader.AddVariable("yj2_dr",'F')
  dataloader.AddVariable("llyjj_dphi",'F')
  dataloader.AddVariable("j1_pt",'F')
  dataloader.AddVariable("j2_pt",'F')
  dataloader.AddVariable("llyjj_ptbal",'F')
  dataloader.AddVariable("yjj_zep",'F')
  dataloader.SetBackgroundWeightExpression("weightXyear")
  dataloader.SetSignalWeightExpression("weightXyear")
  dataloader.AddSpectator("lly_m", 'F')
  dataloader.AddSpectator("weightXyear", 'F')

  # Add data.
  train_chain = TChain('train_tree_nocut')
  train_chain.Add(train_filename)
  test_chain = TChain('eval_tree_nocut')
  test_chain.Add(test_filename)
  dataloader.AddTree(train_chain, 'Background', 1., 'classID==0', TMVA.Types.kTraining)
  dataloader.AddTree(train_chain, 'Signal', 1., 'classID==1', TMVA.Types.kTraining)
  dataloader.AddTree(test_chain, 'Background', 1., 'classID==0', TMVA.Types.kTesting)
  dataloader.AddTree(test_chain, 'Signal', 1., 'classID==1', TMVA.Types.kTesting)
  cut_s = TCut('1');
  cut_b = TCut('1');
  dataloader.PrepareTrainingAndTestTree(cut_s,cut_b,f"NormMode=NumEvents:ScaleWithPreselEff:!V");

  factory.BookMethod(dataloader,TMVA.Types.kBDT,"BDT","!H:!V:NTrees=850:MinNodeSize=2.5%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20");
  factory.TrainAllMethods();
  factory.TestAllMethods();
  factory.EvaluateAllMethods();
  output.Close()

if __name__ == "__main__":
  start_time = time.time()

  input_mva_ntuple = 'ntuples/vbf_mva_ntuples.root'
  bdt_name = 'tmva_bdt_hig19014'
  output_model_folder = f'{bdt_name}_model' # Can't be a subfolder
  output_mva_results = f'mva_output/{bdt_name}_factory_results.root'
  print(f'input_mva_ntuple: {input_mva_ntuple}')
  print(f'bdt_model_folder: {output_model_folder}')
  print(f'output_mva_results: {output_mva_results}')

  train_bdt(model_folder=output_model_folder, 
            train_filename=input_mva_ntuple, 
            test_filename=input_mva_ntuple, 
            tmva_filename=output_mva_results)

  # Run trained bdt over full set
  infer_output_filename = f'mva_output/{bdt_name}_results.root'
  command = f'root -q \'root_scripts/infer_tmva.C+("{input_mva_ntuple}", "train_tree_nocut", "eval_tree_nocut", "test_tree_nocut", "BDT", "{output_model_folder}/weights/TMVAClassification_BDT.weights.xml", "{infer_output_filename}")\''
  print(command)
  os.system(command)

  elapsed_time = time.time() - start_time
  print(f'Training time: {elapsed_time}')

  #TMVA::TMVAGui("ntuples_mva/TMVA_bdt.root")
