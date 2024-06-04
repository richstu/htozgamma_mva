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

def normalize_hist(hist):
  sum_weight = hist.GetSumOfWeights()
  hist.Scale(1/sum_weight)

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
  normalize_hist(base_hist)
  normalize_hist(target_hist)

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

def apply_same_shape_weights(weight_filename, input_filename, nominal_weight_name, shape_weight_name, w_shape_name, apply_class_index, output_filename):
  ROOT.setup_shape_weight(weight_filename)
  ROOT.setup_apply_class_index(apply_class_index)
  defines = [(w_shape_name,'get_shape_weight(lly_m, classID)'),
             (shape_weight_name, f'{nominal_weight_name}*{w_shape_name}')]
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

if __name__ == '__main__':
  start_time = time.time()

  parser = argparse.ArgumentParser(description='Makes ntuples for mva training. Using output from ggf_ntuple_producer.')
  parser.add_argument('-b','--baseline', help='Which baseline to select. tight or hig19014', default='tight')
  args = parser.parse_args()

  name_tag = f'ggfvbf_{args.baseline}'
  if args.baseline == 'tight_012j':
    baseline_cut = '(trigger>0) & (pass_filter>0) & (use_event>0) & (y_id80>0) & (y_pt/lly_m>15./110) & (leplep_m>80) & (leplep_m<100) & (lly_m+leplep_m>185) & (lly_m>100) & (lly_m<180) & (((njet<=1) & (nlep==2) & (met<90)) | ((njet>=2) & (nlep==2) & (nbdfm==0)))'
    train_cut =    '(trigger>0) & (pass_filter>0) & (use_event>0) & (y_id80>0) & (y_pt/lly_m>15./110) & (leplep_m>80) & (leplep_m<100) & (lly_m+leplep_m>185) & (lly_m>100) & (lly_m<180) & (((njet<=1) & (nlep==2) & (met<90)) | ((njet>=2) & (nlep==2) & (nbdfm==0))) & (lly_m>120) & (lly_m<130)'
    decorr_cuts = [] # Used when producing decorr variables.
  elif args.baseline == 'hig19014_012j':
    baseline_cut = '(trigger>0) & (pass_filter>0) & (use_event>0) & (y_pt/lly_m>15./110) & (leplep_m>50) & (lly_m+leplep_m>185) & (lly_m>100) & (lly_m<180) & (((njet<=1) & (nlep==2) & (met<90)) | ((njet>=2) & (nlep==2) & (nbdfm==0)))'
    train_cut =    '(trigger>0) & (pass_filter>0) & (use_event>0) & (y_pt/lly_m>15./110) & (leplep_m>50) & (lly_m+leplep_m>185) & (lly_m>100) & (lly_m<180) & (((njet<=1) & (nlep==2) & (met<90)) | ((njet>=2) & (nlep==2) & (nbdfm==0)))'
    decorr_cuts = [] # Used when producing decorr variables.
  elif args.baseline == 'tight_01j':
    baseline_cut = '(trigger>0) & (pass_filter>0) & (use_event>0) & (y_id80>0) & (y_pt/lly_m>15./110) & (leplep_m>80) & (leplep_m<100) & (lly_m+leplep_m>185) & (lly_m>100) & (lly_m<180) & (njet<=1) & (nlep==2) & (met<90)'
    train_cut =    '(trigger>0) & (pass_filter>0) & (use_event>0) & (y_id80>0) & (y_pt/lly_m>15./110) & (leplep_m>80) & (leplep_m<100) & (lly_m+leplep_m>185) & (lly_m>100) & (lly_m<180) & (njet<=1) & (nlep==2) & (met<90) & (lly_m>120) & (lly_m<130)'
    decorr_cuts = [] # Used when producing decorr variables.
  elif args.baseline == 'hig19014_01j':
    baseline_cut = '(trigger>0) & (pass_filter>0) & (use_event>0) & (y_pt/lly_m>15./110) & (leplep_m>50) & (lly_m+leplep_m>185) & (lly_m>100) & (lly_m<180) & (njet<=1) & (nlep==2) & (met<90)'
    train_cut =    '(trigger>0) & (pass_filter>0) & (use_event>0) & (y_pt/lly_m>15./110) & (leplep_m>50) & (lly_m+leplep_m>185) & (lly_m>100) & (lly_m<180) & (njet<=1) & (nlep==2) & (met<90)'
    decorr_cuts = [] # Used when producing decorr variables.
  elif args.baseline == 'tight_0j':
    baseline_cut = '(trigger>0) & (pass_filter>0) & (use_event>0) & (y_id80>0) & (y_pt/lly_m>15./110) & (leplep_m>80) & (leplep_m<100) & (lly_m+leplep_m>185) & (lly_m>100) & (lly_m<180) & (njet==0) & (nlep==2) & (met<90)'
    train_cut =    '(trigger>0) & (pass_filter>0) & (use_event>0) & (y_id80>0) & (y_pt/lly_m>15./110) & (leplep_m>80) & (leplep_m<100) & (lly_m+leplep_m>185) & (lly_m>100) & (lly_m<180) & (njet==0) & (nlep==2) & (met<90) & (lly_m>120) & (lly_m<130)'
    decorr_cuts = [] # Used when producing decorr variables.
  elif args.baseline == 'tight_1j':
    baseline_cut = '(trigger>0) & (pass_filter>0) & (use_event>0) & (y_id80>0) & (y_pt/lly_m>15./110) & (leplep_m>80) & (leplep_m<100) & (lly_m+leplep_m>185) & (lly_m>100) & (lly_m<180) & (njet==1) & (nlep==2) & (met<90)'
    train_cut =    '(trigger>0) & (pass_filter>0) & (use_event>0) & (y_id80>0) & (y_pt/lly_m>15./110) & (leplep_m>80) & (leplep_m<100) & (lly_m+leplep_m>185) & (lly_m>100) & (lly_m<180) & (njet==1) & (nlep==2) & (met<90) & (lly_m>120) & (lly_m<130)'
    decorr_cuts = [] # Used when producing decorr variables.
  elif args.baseline == 'tight_2j':
    baseline_cut = '(trigger>0) & (pass_filter>0) & (use_event>0) & (y_id80>0) & (y_pt/lly_m>15./110) & (leplep_m>80) & (leplep_m<100) & (lly_m+leplep_m>185) & (lly_m>100) & (lly_m<180) & (njet>=2) & (nlep==2) & (nbdfm==0)'
    train_cut =    '(trigger>0) & (pass_filter>0) & (use_event>0) & (y_id80>0) & (y_pt/lly_m>15./110) & (leplep_m>80) & (leplep_m<100) & (lly_m+leplep_m>185) & (lly_m>100) & (lly_m<180) & (njet>=2) & (nlep==2) & (nbdfm==0) & (lly_m>120) & (lly_m<130)'
    decorr_cuts = [] # Used when producing decorr variables.
  else:
    print(f'[Error] No baseline called {args.baseline}')
    sys.exit()

  np.random.seed(1)

  # Combine signal and background samples into one ntuple AND mix up signal and background event sequence.
  # classID, sampleID, sample_filename
  class_filenames = [
               [0, 1, 'ntuples/ggfvbf_ntuples_ZGToLLG_01J_5f_lowMLL_lowGPt_TuneCP5_13TeV-amcatnloFXFX-pythia8_2016APV201620172018.root'],
               [0, 2, 'ntuples/ggfvbf_ntuples_DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_2016APV201620172018.root'],
               [0, 3, 'ntuples/ggfvbf_ntuples_ZGamma2JToGamma2L2J_EWK_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8_2016APV201620172018.root'],
               [1, 4, 'ntuples/ggfvbf_ntuples_GluGluHToZG_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8_2016APV201620172018.root'],
               [1, 5, 'ntuples/ggfvbf_ntuples_VBFHToZG_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8_2016APV201620172018.root'],
               ]
  sampleIDs = [1,2,3,4,5]
  combined_root_name = f'ntuples/ggfvbf_combined_sample.root'
  if not os.path.isfile(combined_root_name):
    ntuple, branches = mva_ntuple_producer_tools.combine_ntuples(class_filenames)
    # Create combined ntuple
    with uproot.recreate(combined_root_name) as output_file:
      output_file['tree'] = ntuple
    print(f'Created {combined_root_name}')

  # Apply baseline cut and then split sample into training, validation, and testing tree using event number. 
  mva_ntuple_producer_tools.split_ntuple(combined_root_name, 'tree', baseline_cut, train_cut,
    sampleID_branch_name = 'sampleID',
    sampleIDs = sampleIDs,
    weight_branch_names=['weightXyear','w_lumiXyear'], 
    output_filename = f'ntuples/{name_tag}_ntuples.root')
  print(f'Created ntuples/{name_tag}_ntuples.root')

  print('Making lly_m same shape weights branch')
  init_same_shape_weights_functions()
  # Sync background to signal shape
  get_same_shape_weights(f'ntuples/{name_tag}_ntuples.root', 'train_tree', 'lly_m', 'weightXyear', 'classID==1',
                         f'ntuples/{name_tag}_ntuples.root', 'train_tree', 'lly_m', 'weightXyear', 'classID==0',
                         f'ntuples/{name_tag}_btos_shape_weight.root')
  apply_same_shape_weights(f'ntuples/{name_tag}_btos_shape_weight.root', 
                           f'ntuples/{name_tag}_ntuples.root',
                           'weightXyear', 'weightXyearXbtos_shape', 'w_same_btos_shape', apply_class_index=0, output_filename=f'ntuples/{name_tag}_shape_ntuples.root')

  # Make decorrelation variables. Later move to ggf_ntuple_producer.py
  mva_ntuple_producer_tools.init_apply_decorr_func()
  decorr_variables = ['y_pt']
  decorr_defines = [('cosThetamass3','float(cosTheta*(125./lly_m)*(125./lly_m)*(125./lly_m))'),
                    ('cosThetamass2','float(cosTheta*(125./lly_m)*(125./lly_m))'),
                    ('cosThetamass','float(cosTheta*(125./lly_m))'),
                    ('cosThetamass3_abs', 'float(abs(cosTheta*(125./lly_m)*(125./lly_m)*(125./lly_m)))'),
                    ('costheta_abs', 'abs(costheta)'),
                    ('l1_eta_abs', 'abs(l1_eta)'),
                    ('l2_eta_abs', 'abs(l2_eta)'),
                    ('phi_abs', 'abs(phi)'),
                    ('y_eta_abs', 'abs(y_eta)'),
                    ('l1_ptmass', 'l1_pt/lly_m*125'),
                    ('yl_drmaxmass', 'yl_drmax/lly_m*125'),
                    ('ll_lly_m', 'lly_m+leplep_m'),
                    ('y_ptmass2', 'y_pt/lly_m/lly_m'),
                    ('y_ptmass3', 'y_pt/lly_m/lly_m/lly_m'),
                    ('y_ptmass4', 'y_pt/lly_m/lly_m/lly_m/lly_m'),
                    ('y_res0', 'y_res*y_pt'),
                    ('y_ptxmass', 'y_pt*lly_m'),
                    ('y_ptxmass2', 'y_pt*lly_m*lly_m'),
                    ('y_ptcal1', 'get_y_ptcal1(leplep_pt, leplep_eta, leplep_phi, leplep_m, y_pt, y_eta, y_phi)'),
                    ('y_ptcal2', 'get_y_ptcal2(leplep_pt, leplep_eta, leplep_phi, leplep_m, y_pt, y_eta, y_phi)'),
                    ('y_ptcal3', 'get_y_ptcal3(leplep_pt, leplep_eta, leplep_phi, leplep_m, y_pt, y_eta, y_phi, lly_m)'),
                    ('y_ptcal4', 'get_y_ptcal4(leplep_pt, leplep_eta, leplep_phi, leplep_m, y_pt, y_eta, y_phi, lly_m)'),
                    ('yz_mt', 'get_mt(leplep_pt, leplep_eta, leplep_phi, leplep_m, y_pt, y_eta, y_phi, lly_m)'),
                    ('y_ptcal4pt', 'y_ptcal4/y_pt'),
                    ('yll_angle', 'get_yll_angle(leplep_pt, leplep_eta, leplep_phi, leplep_m, y_pt, y_eta, y_phi)'),
                    ('leplep_e', 'get_leplep_e(leplep_pt, leplep_eta, leplep_phi, leplep_m)'),
                    ('leplep_p', 'get_leplep_p(leplep_pt, leplep_eta, leplep_phi, leplep_m)'),
                    ('y_ptrep', '(lly_m*lly_m-leplep_m*leplep_m)/(y_pt*2*cosh(leplep_eta))'),
                    ('y_ptrepmass', '(lly_m*lly_m-leplep_m*leplep_m)/(y_pt*2*cosh(leplep_eta))/lly_m'),
                    ('y_ptinfo', '(lly_m*lly_m-leplep_m*leplep_m)/y_pt'),
                    ('y_ptinfomass', '(lly_m*lly_m-leplep_m*leplep_m)/y_pt/lly_m'),
                    ('y_ptinfo2', 'y_pt*2*cosh(y_eta)'),
                    ('y_ptinfo2mass', 'y_pt*2*cosh(y_eta)/lly_m'),
                    ('y_ptinfo2mass2', 'y_pt*2*cosh(y_eta)/lly_m/lly_m'),
                    ('y_ptinfo2mass22', 'y_pt*2*cosh(y_eta)/((lly_m*lly_m)-(leplep_m*leplep_m))'),
                    ('y_ptinfo2mass3', 'y_pt*2*cosh(y_eta)/lly_m/lly_m/lly_m'),
                    ('y_ptinfo2p1', 'y_pt*2*cosh(y_eta)/(lly_m/lly_m-leplep_m*leplep_m)'),
                    ('y_ptinfo3', 'y_pt*2*cosh(y_eta)*(leplep_e-leplep_p*cos(yll_angle))'),
                   ]
  # Create ntuples with decorrelated variables
  decorr_func_filename = f'ntuples/{name_tag}_decorr_func.root'
  mva_ntuple_producer_tools.get_decorr_func(f'ntuples/{name_tag}_shape_ntuples.root', 'train_tree_baseline', 'lly_m', decorr_variables,'weightXyear', 'classID==0', decorr_func_filename)
  mva_ntuple_producer_tools.apply_decorr_func(decorr_variables, decorr_func_filename, decorr_defines, decorr_cuts,
                    input_filename=f'ntuples/{name_tag}_shape_ntuples.root', output_filename=f'ntuples/{name_tag}_shape_decorr_ntuples.root')

  print('Make training weights')
  input_root_name = f'ntuples/{name_tag}_shape_decorr_ntuples.root'
  output_root_name = f'ntuples/{name_tag}_shape_decorr_weight_ntuples.root'
  classID_name = 'classID'
  weight_name = 'weightXyear'
  sigres_name = 'lly_res'
  output_ntuple = {}
  output_file = uproot.recreate(output_root_name)
  with uproot.open(input_root_name) as root_file:
    tree_names = [str(key.split(';')[0]) for key in root_file.keys()]
    for tree_name in tree_names:
      ntuple_tree = root_file[tree_name]
      branches = [branch for branch in ntuple_tree.keys()]
      ntuple = ntuple_tree.arrays(library='np') # ntuple[branch]: [[value, ..]]
      event_ntuple = np.stack([ntuple[branch] for branch in branches], axis=1)
      classID_index = branches.index(classID_name)
      weight_index = branches.index(weight_name)
      sigres_index = branches.index(sigres_name)
      # Add weights
      # wgteq1
      nbkg = np.count_nonzero(event_ntuple[:,classID_index]==0)
      nsig = np.count_nonzero(event_ntuple[:,classID_index]==1)
      nsig_nbkg = nsig * 1./ nbkg
      wgteq1_array = np.array(event_ntuple[:,classID_index])
      wgteq1_array[wgteq1_array == 0] = nsig_nbkg
      #wgteq1_array = np.expand_dims(wgteq1_array, axis=0)
      ntuple['wgteq1'] = wgteq1_array
      # wgt
      wgt_array = np.array(event_ntuple[:,weight_index])
      wgt_array[wgt_array<0] = 0 # Set negative weights to 0
      #wgt_array = np.expand_dims(wgt_array, axis=0)
      ntuple['wgt'] = wgt_array
      # Ignore negative weights for below calculations
      event_ntuple[event_ntuple[:,weight_index]<0,weight_index] = 0
      # wgt11
      bkg_mask = event_ntuple[:,classID_index]==0
      bkg_sum = np.sum(event_ntuple[bkg_mask,weight_index])
      sig_mask = event_ntuple[:,classID_index]==1
      sig_sum = np.sum(event_ntuple[sig_mask,weight_index])
      bkg_over_sig = bkg_sum/sig_sum
      # Scale signal to match background
      wgt11_array = np.array(event_ntuple[:,weight_index])
      wgt11_array[sig_mask] = wgt11_array[sig_mask] * bkg_over_sig
      wgt11_array[wgt11_array<0] = 0 # Set negative weights to 0
      #wgt11_array = np.expand_dims(wgt11_array, axis=0)
      ntuple['wgt11'] = wgt11_array
      # wgtsigres11
      # Get sum of weights for background and signal
      bkg_sum = np.sum(event_ntuple[bkg_mask,weight_index])
      sig_sum = np.sum(event_ntuple[sig_mask,weight_index]/event_ntuple[sig_mask,sigres_index])
      bkg_over_sig = bkg_sum/sig_sum
      # Scale signal to match background
      wgtsigres11_array = np.array(event_ntuple[:,weight_index])
      wgtsigres11_array[sig_mask] = event_ntuple[sig_mask,weight_index]/event_ntuple[sig_mask,sigres_index] * bkg_over_sig
      wgtsigres11_array[wgtsigres11_array<0] = 0 # Set negative weights to 0
      #wgtsigres11_array = np.expand_dims(wgtsigres11_array, axis=0)
      ntuple['wgtsigres11'] = wgtsigres11_array
      # Write to file
      output_file[tree_name] = ntuple
      #print(tree_name, bkg_sum, sig_sum, bkg_over_sig, ntuple['wgt11'], ntuple['weightXyear'])
      #print(tree_name, ntuple)
      print(f'Created {output_root_name}')

  ## Evaluate nominal features (Group by 6 features)
  #features = ['y_mva', 'costheta','phi', 'y_eta', 'y_res0', 'y_res', 
  #  'yl_drmin', 'yl_drminmass','yl_drmax', 'yl_drmaxmass', 'l1_eta','l2_eta', 
  #  'cosTheta', 'cosThetamass', 'cosThetamass2', 'cosThetamass3', 'lly_pt', 'lly_ptmass', 
  #  'y_pt', 'y_ptmass', 'y_pt_decorr', 'y_ptxmass', 'y_ptxmass2', 'l2_pt',
  #  'l1_pt', 'l1_ptmass', 'leplep_m', 'leplep_pt', 'lly_ptt', 'tru_leplep_m'
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

  ## Evaluate baseline cut (Group by 3 features)
  #features = ['leplep_m', 'll_lly_m', 'y_ptmass',
  #            'y_ptmass2', 'y_ptmass3', 'y_ptmass4',
  #            'y_ptcal1', 'y_ptcal2', 'y_ptcal4',
  #            'y_ptcal4pt', 'yll_angle', 'y_ptrep',
  #            'y_ptrepmass','y_ptinfo', 'y_ptinfo2', 
  #            'y_ptinfo3', 'y_ptinfo2mass', 'y_ptinfomass',
  #            'y_ptinfo2mass2', 'y_ptinfo2mass3', 'y_ptinfo2p1',
  #            'st', 'ht', 'y_ptinfo2mass22']
  #measure_tools.plot_feature_shapes(root_filename=f'ntuples/{name_tag}_decorr_ntuples.root', tree_name='train_tree_baseline', 
  #  observable_name = 'lly_m', feature_names = features, weight_name = 'weightXyear', 
  #  split_cut = 'classID==1', observable_cut = 'classID==0', filename_tag = f'{name_tag}_decorr_ntuples')

  ## Collect pdfs into a file
  #for ipage in range(int(math.ceil(len(features)/3))):
  #  input_branches = ''
  #  for ibranch, input_branch in enumerate(features): 
  #    if ibranch >= ipage*3 and ibranch < (ipage+1)*3: input_branches += f'plots/{name_tag}_decorr_ntuples_{measure_tools.slugify(input_branch)}_bins.pdf plots/{name_tag}_decorr_ntuples_lly_m_{measure_tools.slugify(input_branch)}.pdf '
  #  command = f'pdf_combine.py -f -i {input_branches} -x 2 -y 3 -o plots/{name_tag}_decorr_ntuples_baseline_inputs_{ipage}.pdf'
  #  os.system(command)

  elapsed_time = time.time() - start_time
  print(f'Elapsed time: {elapsed_time:.1f} sec')
