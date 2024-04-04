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

ROOT.gInterpreter.Declare("""
template <class C>
using RVec = ROOT::VecOps::RVec<C>;

float get_y_ptcal1(float z_pt, float z_eta, float z_phi, float z_m, float y_pt, float y_eta, float y_phi) {
  TLorentzVector zboson;
  zboson.SetPtEtaPhiM(z_pt, z_eta, z_phi, z_m);
  TLorentzVector photon;
  photon.SetPtEtaPhiM(y_pt, y_eta, y_phi, 0);
  float angle = zboson.Angle(photon.Vect());
  float result = (125*125-zboson.M()*zboson.M())/(2*cosh(y_eta)*(zboson.E()-zboson.P()*cos(angle)));
  return result;
}

float get_y_ptcal2(float z_pt, float z_eta, float z_phi, float z_m, float y_pt, float y_eta, float y_phi) {
  TLorentzVector zboson;
  zboson.SetPtEtaPhiM(z_pt, z_eta, z_phi, z_m);
  TLorentzVector photon;
  photon.SetPtEtaPhiM(y_pt, y_eta, y_phi, 0);
  float angle = zboson.Angle(photon.Vect());
  float result = (125*125-91.19*91.19)/(2*cosh(y_eta)*(zboson.E()-zboson.P()*cos(angle)));
  return result;
}

float get_y_ptcal3(float z_pt, float z_eta, float z_phi, float z_m, float y_pt, float y_eta, float y_phi, float h_m) {
  TLorentzVector zboson;
  zboson.SetPtEtaPhiM(z_pt, z_eta, z_phi, z_m);
  TLorentzVector photon;
  photon.SetPtEtaPhiM(y_pt, y_eta, y_phi, 0);
  float angle = zboson.Angle(photon.Vect());
  float result = (h_m*h_m-zboson.M()*zboson.M())/(2*cosh(y_eta)*(zboson.E()-zboson.P()*cos(angle)));
  return result;
}

float get_y_ptcal4(float z_pt, float z_eta, float z_phi, float z_m, float y_pt, float y_eta, float y_phi, float h_m) {
  TLorentzVector zboson;
  zboson.SetPtEtaPhiM(z_pt, z_eta, z_phi, z_m);
  TLorentzVector photon;
  photon.SetPtEtaPhiM(y_pt, y_eta, y_phi, 0);
  float angle = zboson.Angle(photon.Vect());
  float result = (h_m*h_m-91.19*91.19)/(2*cosh(y_eta)*(zboson.E()-zboson.P()*cos(angle)));
  return result;
}

float get_mt(float z_pt, float z_eta, float z_phi, float z_m, float y_pt, float y_eta, float y_phi, float h_m) {
  TLorentzVector zboson;
  zboson.SetPtEtaPhiM(z_pt, z_eta, z_phi, z_m);
  TLorentzVector photon;
  photon.SetPtEtaPhiM(y_pt, y_eta, y_phi, 0);

  float angle = zboson.Angle(photon.Vect());
  float result = 2*z_pt*y_pt*(1-angle);
  return result;
}

float get_yll_angle(float z_pt, float z_eta, float z_phi, float z_m, float y_pt, float y_eta, float y_phi) {
  TLorentzVector zboson;
  zboson.SetPtEtaPhiM(z_pt, z_eta, z_phi, z_m);
  TLorentzVector photon;
  photon.SetPtEtaPhiM(y_pt, y_eta, y_phi, 0);
  float angle = zboson.Angle(photon.Vect());
  return angle;
}

float get_leplep_e(float z_pt, float z_eta, float z_phi, float z_m) {
  TLorentzVector zboson;
  zboson.SetPtEtaPhiM(z_pt, z_eta, z_phi, z_m);
  return zboson.E();
}

float get_leplep_p(float z_pt, float z_eta, float z_phi, float z_m) {
  TLorentzVector zboson;
  zboson.SetPtEtaPhiM(z_pt, z_eta, z_phi, z_m);
  return zboson.P();
}

""")

if __name__ == '__main__':
  start_time = time.time()

  parser = argparse.ArgumentParser(description='Makes ntuples for mva training. Using output from ggf_ntuple_producer.')
  parser.add_argument('-b','--baseline', help='Which baseline to select. tight or hig19014 or hig19014equal.', default='tight')
  args = parser.parse_args()

  if args.baseline == 'tight':
    name_tag = 'ggf_mva'
    baseline_cut = '(trigger>0) & (pass_filter>0) & (use_event>0) & (y_id80>0) & (y_pt/lly_m>15./110) & (leplep_m>80) & (leplep_m<100) & (lly_m+leplep_m>185) & (lly_m>100) & (lly_m<180) & (njet<=1) & (nlep<=2) & (met<90)'
    train_cut =    '(trigger>0) & (pass_filter>0) & (use_event>0) & (y_id80>0) & (y_pt/lly_m>15./110) & (leplep_m>80) & (leplep_m<100) & (lly_m+leplep_m>185) & (lly_m>100) & (lly_m<180) & (njet<=1) & (nlep<=2) & (met<90) & (lly_m>120) & (lly_m<130)'
    decorr_cuts = [] # Used when producing decorr variables.
  elif args.baseline == 'hig19014':
    name_tag = 'ggf_mva_hig19014'
    baseline_cut = '(trigger>0) & (pass_filter>0) & (use_event>0) & (y_pt/lly_m>15./110) & (leplep_m>50) & (lly_m+leplep_m>185) & (lly_m>100) & (lly_m<180) & (njet<=1) & (nlep<=2) & (met<90)'
    train_cut =    '(trigger>0) & (pass_filter>0) & (use_event>0) & (y_pt/lly_m>15./110) & (leplep_m>50) & (lly_m+leplep_m>185) & (lly_m>100) & (lly_m<180) & (njet<=1) & (nlep<=2) & (met<90)'
    decorr_cuts = [] # Used when producing decorr variables.
  elif args.baseline == 'tightloose':
    name_tag = 'ggf_mva_tightloose'
    baseline_cut = '(trigger>0) & (pass_filter>0) & (use_event>0) & (y_id80>0) & (leplep_m>80) & (leplep_m<100) & (lly_m+leplep_m>185) & (lly_m>100) & (lly_m<180) & (njet<=1) & (nlep<=2) & (met<90)'
    train_cut =    '(trigger>0) & (pass_filter>0) & (use_event>0) & (y_id80>0) & (leplep_m>80) & (leplep_m<100) & (lly_m+leplep_m>185) & (lly_m>100) & (lly_m<180) & (njet<=1) & (nlep<=2) & (met<90) & (lly_m>120) & (lly_m<130)'
    decorr_cuts = [] # Used when producing decorr variables.
  elif args.baseline == 'tightyptmass2':
    name_tag = 'ggf_mva_yptmass2'
    baseline_cut = '(trigger>0) & (pass_filter>0) & (use_event>0) & (y_id80>0) & (y_pt/lly_m/lly_m>15./110/110) & (leplep_m>80) & (leplep_m<100) & (lly_m+leplep_m>185) & (lly_m>100) & (lly_m<180) & (njet<=1) & (nlep<=2) & (met<90)'
    train_cut =    '(trigger>0) & (pass_filter>0) & (use_event>0) & (y_id80>0) & (y_pt/lly_m/lly_m>15./110/110) & (leplep_m>80) & (leplep_m<100) & (lly_m+leplep_m>185) & (lly_m>100) & (lly_m<180) & (njet<=1) & (nlep<=2) & (met<90) & (lly_m>120) & (lly_m<130)'
    decorr_cuts = [] # Used when producing decorr variables.
  if args.baseline == 'tightcosTheta':
    name_tag = 'ggf_mva_cosTheta'
    baseline_cut = '(trigger>0) & (pass_filter>0) & (use_event>0) & (y_id80>0) & (y_pt/lly_m>15./110) & (leplep_m>80) & (leplep_m<100) & (lly_m+leplep_m>185) & (lly_m>100) & (lly_m<180) & (cosTheta>-0.7) & (cosTheta<0.7) & (njet<=1) & (nlep<=2) & (met<90)'
    train_cut =    '(trigger>0) & (pass_filter>0) & (use_event>0) & (y_id80>0) & (y_pt/lly_m>15./110) & (leplep_m>80) & (leplep_m<100) & (lly_m+leplep_m>185) & (lly_m>100) & (lly_m<180) & (cosTheta>-0.7) & (cosTheta<0.7) & (njet<=1) & (nlep<=2) & (met<90) & (lly_m>120) & (lly_m<130)'
    decorr_cuts = [] # Used when producing decorr variables.

  np.random.seed(1)

  ## Combine signal and background samples into one ntuple AND mix up signal and background event sequence.
  ## classID, sampleID, sample_filename
  #class_filenames = [
  #             [0, 1, 'ntuples/ggf_ntuples_ZGToLLG_01J_5f_lowMLL_lowGPt_TuneCP5_13TeV-amcatnloFXFX-pythia8_2016APV201620172018.root'],
  #             [0, 2, 'ntuples/ggf_ntuples_DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_2016APV201620172018.root'],
  #             [1, 3, 'ntuples/ggf_ntuples_GluGluHToZG_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8_2016APV201620172018.root'],
  #             ]
  #ntuple, branches = mva_ntuple_producer_tools.combine_ntuples(class_filenames)

  ## Split sample into training, validation, and testing tree using event number. 
  #mva_ntuple_producer_tools.split_ntuple(ntuple, branches, baseline_cut, train_cut,
  #  weight_branch_names=['weightXyear','w_lumiXyear'], 
  #  output_filename = f'ntuples/{name_tag}_ntuples.root')
  #
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
  #                  ('ll_lly_m', 'lly_m+leplep_m'),
  #                  ('y_ptmass2', 'y_pt/lly_m/lly_m'),
  #                  ('y_ptmass3', 'y_pt/lly_m/lly_m/lly_m'),
  #                  ('y_ptmass4', 'y_pt/lly_m/lly_m/lly_m/lly_m'),
  #                  ('y_res0', 'y_res*y_pt'),
  #                  ('y_ptxmass', 'y_pt*lly_m'),
  #                  ('y_ptxmass2', 'y_pt*lly_m*lly_m'),
  #                  ('y_ptcal1', 'get_y_ptcal1(leplep_pt, leplep_eta, leplep_phi, leplep_m, y_pt, y_eta, y_phi)'),
  #                  ('y_ptcal2', 'get_y_ptcal2(leplep_pt, leplep_eta, leplep_phi, leplep_m, y_pt, y_eta, y_phi)'),
  #                  ('y_ptcal3', 'get_y_ptcal3(leplep_pt, leplep_eta, leplep_phi, leplep_m, y_pt, y_eta, y_phi, lly_m)'),
  #                  ('y_ptcal4', 'get_y_ptcal4(leplep_pt, leplep_eta, leplep_phi, leplep_m, y_pt, y_eta, y_phi, lly_m)'),
  #                  ('yz_mt', 'get_mt(leplep_pt, leplep_eta, leplep_phi, leplep_m, y_pt, y_eta, y_phi, lly_m)'),
  #                  ('y_ptcal4pt', 'y_ptcal4/y_pt'),
  #                  ('yll_angle', 'get_yll_angle(leplep_pt, leplep_eta, leplep_phi, leplep_m, y_pt, y_eta, y_phi)'),
  #                  ('leplep_e', 'get_leplep_e(leplep_pt, leplep_eta, leplep_phi, leplep_m)'),
  #                  ('leplep_p', 'get_leplep_p(leplep_pt, leplep_eta, leplep_phi, leplep_m)'),
  #                  ('y_ptrep', '(lly_m*lly_m-leplep_m*leplep_m)/(y_pt*2*cosh(leplep_eta))'),
  #                  ('y_ptrepmass', '(lly_m*lly_m-leplep_m*leplep_m)/(y_pt*2*cosh(leplep_eta))/lly_m'),
  #                  ('y_ptinfo', '(lly_m*lly_m-leplep_m*leplep_m)/y_pt'),
  #                  ('y_ptinfomass', '(lly_m*lly_m-leplep_m*leplep_m)/y_pt/lly_m'),
  #                  ('y_ptinfo2', 'y_pt*2*cosh(y_eta)'),
  #                  ('y_ptinfo2mass', 'y_pt*2*cosh(y_eta)/lly_m'),
  #                  ('y_ptinfo2mass2', 'y_pt*2*cosh(y_eta)/lly_m/lly_m'),
  #                  ('y_ptinfo2mass3', 'y_pt*2*cosh(y_eta)/lly_m/lly_m/lly_m'),
  #                  ('y_ptinfo2p1', 'y_pt*2*cosh(y_eta)/(lly_m/lly_m-leplep_m*leplep_m)'),
  #                  ('y_ptinfo3', 'y_pt*2*cosh(y_eta)*(leplep_e-leplep_p*cos(yll_angle))'),
  #                 ]
  ## Create ntuples with decorrelated variables
  #decorr_func_filename = f'ntuples/{name_tag}_decorr_func.root'
  #mva_ntuple_producer_tools.get_decorr_func(f'ntuples/{name_tag}_ntuples.root', 'train_tree_baseline', 'lly_m', decorr_variables,'weightXyear', 'classID==0', decorr_func_filename)
  #mva_ntuple_producer_tools.apply_decorr_func(decorr_variables, decorr_func_filename, decorr_defines, decorr_cuts,
  #                  input_filename=f'ntuples/{name_tag}_ntuples.root', output_filename=f'ntuples/{name_tag}_decorr_ntuples.root')

  ## Evaluate nominal features (Group by 6 features)
  #features = ['y_mva', 'costheta','phi', 'y_eta', 'y_res0', 'y_res', 
  #  'yl_drmin', 'yl_drminmass','yl_drmax', 'yl_drmaxmass', 'l1_eta','l2_eta', 
  #  'cosTheta', 'cosThetamass', 'cosThetamass2', 'cosThetamass3', 'lly_pt', 'lly_ptmass', 
  #  'y_pt', 'y_ptmass', 'y_pt_decorr', 'y_ptxmass', 'y_ptxmass2', 'l2_pt',
  #  'l1_pt', 'l1_ptmass', 
  #  ]
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

  # Evaluate baseline cut (Group by 3 features)
  features = ['leplep_m', 'll_lly_m', 'y_ptmass',
              'y_ptmass2', 'y_ptmass3', 'y_ptmass4',
              'y_ptcal1', 'y_ptcal2', 'y_ptcal4',
              'y_ptcal4pt', 'yll_angle', 'y_ptrep',
              'y_ptrepmass','y_ptinfo', 'y_ptinfo2', 
              'y_ptinfo3', 'y_ptinfo2mass', 'y_ptinfomass',
              'y_ptinfo2mass2', 'y_ptinfo2mass3', 'y_ptinfo2p1',
              'st', 'ht']
  measure_tools.plot_feature_shapes(root_filename=f'ntuples/{name_tag}_decorr_ntuples.root', tree_name='train_tree_baseline', 
    observable_name = 'lly_m', feature_names = features, weight_name = 'weightXyear', 
    split_cut = 'classID==1', observable_cut = 'classID==0', filename_tag = f'{name_tag}_decorr_ntuples')

  # Collect pdfs into a file
  for ipage in range(int(math.ceil(len(features)/3))):
    input_branches = ''
    for ibranch, input_branch in enumerate(features): 
      if ibranch >= ipage*3 and ibranch < (ipage+1)*3: input_branches += f'plots/{name_tag}_decorr_ntuples_{measure_tools.slugify(input_branch)}_bins.pdf plots/{name_tag}_decorr_ntuples_lly_m_{measure_tools.slugify(input_branch)}.pdf '
    command = f'pdf_combine.py -f -i {input_branches} -x 2 -y 3 -o plots/{name_tag}_decorr_ntuples_baseline_inputs_{ipage}.pdf'
    os.system(command)

  elapsed_time = time.time() - start_time
  print(f'Elapsed time: {elapsed_time}')
