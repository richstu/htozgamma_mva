#!/usr/bin/env python3
import uproot
import numpy as np
import os
import time
import ROOT
import array
import math
import argparse

import sys
sys.path.append('../python_scripts')
import mva_ntuple_producer_tools
import measure_tools

ROOT.gSystem.AddIncludePath(f'-I{os.environ["WORK_DIR"]}/root_scripts')
ROOT.gInterpreter.ProcessLine('.L infer_tmva.C+')
ROOT.gInterpreter.Declare("""
template <class C>
using RVec = ROOT::VecOps::RVec<C>;

infer_tmva_c infer_ggf("BDT", "ggf_standard_tmva_bdt_hig19014_model/weights/TMVAClassification_BDT.weights.xml");

float get_ggf_bdt(float y_mva, float yl_drmin, float yl_drmax, float lly_ptmass, float cosTheta, float costheta, float phi, float y_res, float y_eta, float l1_eta, float l2_eta) {
  vector<double> input = {y_mva, yl_drmin, yl_drmax, lly_ptmass, cosTheta, costheta, phi, y_res, y_eta, l1_eta, l2_eta};
  return infer_ggf.infer(input);
}
""")

if __name__ == '__main__':
  start_time = time.time()

  parser = argparse.ArgumentParser(description='Makes ntuples for mva training. Using output from ggf_ntuple_producer.')
  parser.add_argument('-b','--baseline', help='Which baseline to select. tight or hig19014.', default='tight')
  args = parser.parse_args()
  np.random.seed(1)

  if args.baseline == 'tight':
    name_tag = 'vbf_mva'
    baseline_cut = '(trigger>0) & (pass_filter>0) & (use_event>0) & (y_id80>0) & (y_pt/lly_m>15./110) & (leplep_m>80) & (leplep_m<100) & (lly_m+leplep_m>185) & (lly_m>100) & (lly_m<180) & (njet>=2) & (nlep==2) & (nbdfm==0)'
    train_cut =    '(trigger>0) & (pass_filter>0) & (use_event>0) & (y_id80>0) & (y_pt/lly_m>15./110) & (leplep_m>80) & (leplep_m<100) & (lly_m+leplep_m>185) & (lly_m>100) & (lly_m<180) & (njet>=2) & (nlep==2) & (nbdfm==0) & (lly_m>120) & (lly_m<130)'
    decorr_cuts = [] # Used when producing decorr variables.
  elif args.baseline == 'hig19014':
    name_tag = 'vbf_mva_hig19014'
    baseline_cut = '(trigger>0) & (pass_filter>0) & (use_event>0) & (y_pt/lly_m>15./110) & (leplep_m>50) & (lly_m+leplep_m>185) & (lly_m>100) & (lly_m<180) & (njet>=2) & (nlep==2) & (nbdfm==0)'
    train_cut =    '(trigger>0) & (pass_filter>0) & (use_event>0) & (y_pt/lly_m>15./110) & (leplep_m>50) & (lly_m+leplep_m>185) & (lly_m>100) & (lly_m<180) & (njet>=2) & (nlep==2) & (nbdfm==0)'
    decorr_cuts = [] # Used when producing decorr variables.

  ## Combine signal and background samples into one ntuple AND mix up signal and background event sequence.
  ## classID, sampleID, sample_filename
  #class_filenames = [
  #             [0, 1, 'ntuples/vbf_ntuples_ZGToLLG_01J_5f_lowMLL_lowGPt_TuneCP5_13TeV-amcatnloFXFX-pythia8_2016APV201620172018.root'],
  #             [0, 2, 'ntuples/vbf_ntuples_DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_2016APV201620172018.root'],
  #             [0, 3, 'ntuples/vbf_ntuples_ZGamma2JToGamma2L2J_EWK_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8_2016APV201620172018.root'],
  #             [1, 4, 'ntuples/vbf_ntuples_GluGluHToZG_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8_2016APV201620172018.root'],
  #             [1, 5, 'ntuples/vbf_ntuples_VBFHToZG_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8_2016APV201620172018.root'],
  #             ]
  #ntuple, branches = mva_ntuple_producer_tools.combine_ntuples(class_filenames)
  ## Create combined ntuple
  #with uproot.recreate(f'ntuples/{name_tag}_combined_sample.root') as output_file:
  #  output_file['tree'] = ntuple
  #print(f'Created ntuples/{name_tag}_combined_sample.root')

  ## Apply baseline cut and then split sample into training, validation, and testing tree using event number. 
  #mva_ntuple_producer_tools.split_ntuple(f'ntuples/{name_tag}_combined_sample.root', 'tree', baseline_cut, train_cut,
  #  sampleID_branch_name = 'sampleID',
  #  sampleIDs = [1,2,3,4,5],
  #  weight_branch_names=['weightXyear','w_lumiXyear'], 
  #  output_filename = f'ntuples/{name_tag}_ntuples.root')
  #print(f'Created ntuples/{name_tag}_ntuples.root')

  ## Make decorrelation variables. Later move to ggf_ntuple_producer.py
  #mva_ntuple_producer_tools.init_apply_decorr_func()
  #decorr_variables = ['y_pt']
  #decorr_defines = [('cosThetamass3','float(cosTheta*(125./lly_m)*(125./lly_m)*(125./lly_m))'),
  #                  ('cosThetamass2','float(cosTheta*(125./lly_m)*(125./lly_m))'),
  #                  ('cosThetamass','float(cosTheta*(125./lly_m))'),
  #                  ('cosThetamass3_abs', 'float(abs(cosTheta*(125./lly_m)*(125./lly_m)*(125./lly_m)))'),
  #                  ('costheta_abs', 'abs(costheta)'),
  #                  ('l1_eta_abs', 'abs(l1_eta)'),
  #                  ('l2_eta_abs', 'abs(l2_eta)'),
  #                  ('phi_abs', 'abs(phi)'),
  #                  ('y_eta_abs', 'abs(y_eta)'),
  #                  ('l1_ptmass', 'l1_pt/lly_m*125'),
  #                  ('yl_drmaxmass', 'yl_drmax/lly_m*125'),
  #                  ('y_ptinfo2mass2', 'y_pt*2*cosh(y_eta)/lly_m/lly_m'),
  #                  ('ggf_bdt', 'get_ggf_bdt(y_mva, yl_drmin, yl_drmax, lly_ptmass, cosTheta, costheta, phi, y_res, y_eta, l1_eta, l2_eta)'),
  #                 ]
  ## Create ntuples with decorrelated variables
  #decorr_func_filename = f'ntuples/{name_tag}_decorr_func.root'
  #mva_ntuple_producer_tools.get_decorr_func(f'ntuples/{name_tag}_ntuples.root', 'train_tree_baseline', 'lly_m', decorr_variables,'weightXyear', 'classID==0', decorr_func_filename)
  #mva_ntuple_producer_tools.apply_decorr_func(decorr_variables, decorr_func_filename, decorr_defines, decorr_cuts,
  #                  input_filename=f'ntuples/{name_tag}_ntuples.root', output_filename=f'ntuples/{name_tag}_decorr_ntuples.root')

  ## Apply baseline cut and then split sample into training, validation using event number. 
  #mva_ntuple_producer_tools.split_ntuple_train_eval(f'ntuples/{name_tag}_combined_sample.root', 'tree', baseline_cut, train_cut,
  #  sampleID_branch_name = 'sampleID',
  #  sampleIDs = [1,2,3,4,5],
  #  weight_branch_names=['weightXyear','w_lumiXyear'], 
  #  output_filename = f'ntuples/{name_tag}_trainvalidation_ntuples.root')
  #print(f'Created ntuples/{name_tag}_trainvalidation_ntuples.root')

  ## Make decorrelation variables. Later move to ggf_ntuple_producer.py
  #mva_ntuple_producer_tools.init_apply_decorr_func()
  #decorr_variables = ['y_pt']
  #decorr_defines = [('cosThetamass3','float(cosTheta*(125./lly_m)*(125./lly_m)*(125./lly_m))'),
  #                  ('cosThetamass2','float(cosTheta*(125./lly_m)*(125./lly_m))'),
  #                  ('cosThetamass','float(cosTheta*(125./lly_m))'),
  #                  ('cosThetamass3_abs', 'float(abs(cosTheta*(125./lly_m)*(125./lly_m)*(125./lly_m)))'),
  #                  ('costheta_abs', 'abs(costheta)'),
  #                  ('l1_eta_abs', 'abs(l1_eta)'),
  #                  ('l2_eta_abs', 'abs(l2_eta)'),
  #                  ('phi_abs', 'abs(phi)'),
  #                  ('y_eta_abs', 'abs(y_eta)'),
  #                  ('l1_ptmass', 'l1_pt/lly_m*125'),
  #                  ('yl_drmaxmass', 'yl_drmax/lly_m*125'),
  #                  ('y_ptinfo2mass2', 'y_pt*2*cosh(y_eta)/lly_m/lly_m'),
  #                  ('ggf_bdt', 'get_ggf_bdt(y_mva, yl_drmin, yl_drmax, lly_ptmass, cosTheta, costheta, phi, y_res, y_eta, l1_eta, l2_eta)'),
  #                 ]
  ## Create ntuples with decorrelated variables
  #decorr_func_filename = f'ntuples/{name_tag}_trainvalidation_decorr_func.root'
  #mva_ntuple_producer_tools.get_decorr_func(f'ntuples/{name_tag}_trainvalidation_ntuples.root', 'train_tree_baseline', 'lly_m', decorr_variables,'weightXyear', 'classID==0', decorr_func_filename)
  #mva_ntuple_producer_tools.apply_decorr_func(decorr_variables, decorr_func_filename, decorr_defines, decorr_cuts,
  #                  input_filename=f'ntuples/{name_tag}_trainvalidation_ntuples.root', output_filename=f'ntuples/{name_tag}_trainvalidation_decorr_ntuples.root')


  ## Evaluate nominal features (Group by 6 features)
  #features = ['y_mva', 'y_res', 'yl_drmin', 'yl_drminmass','yl_drmax', 'yl_drmaxmass',
  #  'cosTheta', 'cosThetamass', 'cosThetamass2', 'cosThetamass3','costheta','phi', 
  #  'lly_pt', 'lly_ptmass', 'lly_ptt', 'y_eta', 'l1_eta', 'l2_eta', 
  #  'y_pt', 'y_ptmass', 'y_pt_decorr', 'l1_pt', 'l1_ptmass', 'l2_pt',
  #  'jj_deta', 'jj_dphi', 'yj1_dr', 'yj2_dr', 'j1_pt', 'j2_pt', 
  #  'llyjj_ptbal', 'yjj_zep', 'llyjj_dphi', ]
  #measure_tools.plot_feature_shapes(root_filename=f'ntuples/{name_tag}_decorr_ntuples.root', tree_name='train_tree_baseline', 
  #  observable_name = 'lly_m', feature_names = features, weight_name = 'weightXyear', 
  #  split_cut = 'classID==1', observable_cut = 'classID==0', filename_tag = f'{name_tag}_decorr_ntuples')

  ## Collect pdfs into a file
  #for ipage in range(int(math.ceil(len(features)/6))):
  #  input_branches = ''
  #  for ibranch, input_branch in enumerate(features): 
  #    if ibranch >= ipage*6 and ibranch < (ipage+1)*6: input_branches += f'plots/{name_tag}_decorr_ntuples_{measure_tools.slugify(input_branch)}_bins.pdf plots/{name_tag}_decorr_ntuples_lly_m_{measure_tools.slugify(input_branch)}.pdf '
  #  command = f'pdf_combine.py -f -i {input_branches} -x 4 -y 3 -o plots/{name_tag}_decorr_ntuples_inputs_{ipage}.pdf'
  #  os.system(command)

  # Evaluate nominal features (Group by 2 features)
  features = ['yl_drmin', 'yl_drminmass',
    'cosTheta', 'cosThetamass', 
  ]
  measure_tools.plot_feature_shapes(root_filename=f'ntuples/{name_tag}_decorr_ntuples.root', tree_name='train_tree_baseline', 
    observable_name = 'lly_m', feature_names = features, weight_name = 'weightXyear', 
    split_cut = 'classID==1', observable_cut = 'classID==0', filename_tag = f'{name_tag}_decorr_ntuples')

  # Collect pdfs into a file
  nfeatures = 2
  for ipage in range(int(math.ceil(len(features)/nfeatures))):
    input_branches = ''
    for ibranch, input_branch in enumerate(features): 
      if ibranch >= ipage*nfeatures and ibranch < (ipage+1)*nfeatures: input_branches += f'plots/{name_tag}_decorr_ntuples_{measure_tools.slugify(input_branch)}_bins.pdf plots/{name_tag}_decorr_ntuples_lly_m_{measure_tools.slugify(input_branch)}.pdf '
    command = f'pdf_combine.py -f -i {input_branches} -x 2 -y 2 -o plots/{name_tag}_decorr_ntuples_inputs_{ipage}.pdf'
    os.system(command)

  elapsed_time = time.time() - start_time
  print(f'Elapsed time: {elapsed_time:.1f} sec')
