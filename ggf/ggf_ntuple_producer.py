#!/usr/bin/env python3
'''
Function that generates n-tuples for MVA training
'''
import ROOT
import time

def write_ntuples(filenames, cuts, out_name, defines=[], tree_name='tree', branches=()):
  '''Generate ROOT n-tuple from existing n-tuple
  
  Parameters:
  filenames - list of filenames of signal ROOT n-tuples
  cuts - list of cuts expressed as strings in order they should be applied
  out_name - output filename of n-tuple
  defines - list of 2-tuples describing new branches to define in the format (name, expr)
            note that these must be defineable before cuts
  tree_name - name of tree in ROOT file
  branches - tuple of branches to save; if empty all branches are saved
  '''
  filenames_vec = ROOT.std.vector('string')()
  #print(filenames)
  for filename in filenames:
    filenames_vec.push_back(filename)
  df = ROOT.RDataFrame('tree',filenames_vec)
  df = df.DefinePerSample("luminosity", "get_luminosity(rdfslot_, rdfsampleinfo_)")
  df = df.DefinePerSample("year", "get_year(rdfslot_, rdfsampleinfo_)")
  df = df.Define("w_lumiXyear", "w_lumi * luminosity")
  df = df.Define("weightXyear", "w_lumi * w_lep * w_bhig_df * w_isr * w_pu * w_prefire * w_photon * luminosity")
  for define in defines:
    df = df.Define(define[0],define[1])
  for cut in cuts:
    df = df.Filter(cut)
  if (branches == ()):
    df.Snapshot(tree_name,'./ntuples/'+out_name)
  else:
    df.Snapshot(tree_name,'./ntuples/'+out_name,branches)
  print('Wrote ntuples/'+out_name)

ROOT.gInterpreter.Declare("""
template <class C>
using RVec = ROOT::VecOps::RVec<C>;

float get_dr(float eta1, float phi1, float eta2, float phi2) {
  const double PI = 3.1415;
  double dphi = fmod(fabs(phi2-phi1), 2.*PI);
  dphi = dphi>PI ? 2.*PI-dphi : dphi;
  double deta = fabs(eta1-eta2);
  return sqrt(deta*deta+dphi*dphi);
}

float get_max_dr(RVec<float> photon_eta, RVec<float> photon_phi, 
    RVec<float> el_eta, RVec<float> el_phi, RVec<float> mu_eta,
    RVec<float> mu_phi, RVec<int> ll_lepid, RVec<int> ll_i1,
    RVec<int> ll_i2) {
  float dr1, dr2;
  if (ll_lepid[0]==11) {
    dr1 = get_dr(photon_eta[0],photon_phi[0],el_eta[ll_i1[0]],el_phi[ll_i1[0]]);
    dr2 = get_dr(photon_eta[0],photon_phi[0],el_eta[ll_i2[0]],el_phi[ll_i2[0]]);
    return dr1 > dr2 ? dr1 : dr2;
  }
  dr1 = get_dr(photon_eta[0],photon_phi[0],mu_eta[ll_i1[0]],mu_phi[ll_i1[0]]);
  dr2 = get_dr(photon_eta[0],photon_phi[0],mu_eta[ll_i2[0]],mu_phi[ll_i2[0]]);
  return dr1 > dr2 ? dr1 : dr2;
}

float get_l1_eta(RVec<float> el_pt, RVec<float> el_eta, 
    RVec<float> mu_pt, RVec<float> mu_eta, RVec<int> ll_lepid, 
    RVec<int> ll_i1, RVec<int> ll_i2) {
  if (ll_lepid[0]==11) {
    return (el_pt[ll_i1[0]] > el_pt[ll_i2[0]]) ? el_eta[ll_i1[0]] : el_eta[ll_i2[0]];
  }
  return (mu_pt[ll_i1[0]] > mu_pt[ll_i2[0]]) ? mu_eta[ll_i1[0]] : mu_eta[ll_i2[0]];
}

float get_l2_eta(RVec<float> el_pt, RVec<float> el_eta, 
    RVec<float> mu_pt, RVec<float> mu_eta, RVec<int> ll_lepid, 
    RVec<int> ll_i1, RVec<int> ll_i2) {
  if (ll_lepid[0]==11) {
    return (el_pt[ll_i1[0]] > el_pt[ll_i2[0]]) ? el_eta[ll_i2[0]] : el_eta[ll_i1[0]];
  }
  return (mu_pt[ll_i1[0]] > mu_pt[ll_i2[0]]) ? mu_eta[ll_i2[0]] : mu_eta[ll_i1[0]];
}

float get_l1_pt(RVec<float> el_pt,
    RVec<float> mu_pt, RVec<int> ll_lepid, 
    RVec<int> ll_i1, RVec<int> ll_i2) {
  if (ll_lepid[0]==11) {
    return (el_pt[ll_i1[0]] > el_pt[ll_i2[0]]) ? el_pt[ll_i1[0]] : el_pt[ll_i2[0]];
  }
  return (mu_pt[ll_i1[0]] > mu_pt[ll_i2[0]]) ? mu_pt[ll_i1[0]] : mu_pt[ll_i2[0]];
}

float get_l2_pt(RVec<float> el_pt,
    RVec<float> mu_pt, RVec<int> ll_lepid, 
    RVec<int> ll_i1, RVec<int> ll_i2) {
  if (ll_lepid[0]==11) {
    return (el_pt[ll_i1[0]] > el_pt[ll_i2[0]]) ? el_pt[ll_i2[0]] : el_pt[ll_i1[0]];
  }
  return (mu_pt[ll_i1[0]] > mu_pt[ll_i2[0]]) ? mu_pt[ll_i2[0]] : mu_pt[ll_i1[0]];
}

float get_l1_phi(RVec<float> el_pt, RVec<float> el_phi, 
    RVec<float> mu_pt, RVec<float> mu_phi, RVec<int> ll_lepid, 
    RVec<int> ll_i1, RVec<int> ll_i2) {
  if (ll_lepid[0]==11) {
    return (el_pt[ll_i1[0]] > el_pt[ll_i2[0]]) ? el_phi[ll_i1[0]] : el_phi[ll_i2[0]];
  }
  return (mu_pt[ll_i1[0]] > mu_pt[ll_i2[0]]) ? mu_phi[ll_i1[0]] : mu_phi[ll_i2[0]];
}

float get_l2_phi(RVec<float> el_pt, RVec<float> el_phi, 
    RVec<float> mu_pt, RVec<float> mu_phi, RVec<int> ll_lepid, 
    RVec<int> ll_i1, RVec<int> ll_i2) {
  if (ll_lepid[0]==11) {
    return (el_pt[ll_i1[0]] > el_pt[ll_i2[0]]) ? el_phi[ll_i2[0]] : el_phi[ll_i1[0]];
  }
  return (mu_pt[ll_i1[0]] > mu_pt[ll_i2[0]]) ? mu_phi[ll_i2[0]] : mu_phi[ll_i1[0]];
}

float H_t(RVec<float> photon_pt,RVec<float> photon_eta,RVec<float> photon_phi,
          RVec<float> el_pt,RVec<float> el_eta,RVec<float> el_phi,
          RVec<float> mu_pt,RVec<float> mu_eta,RVec<float> mu_phi,RVec<int> ll_lepid,RVec<int> ll_i1,RVec<int> ll_i2,
          RVec<float> jet_pt,RVec<float> jet_eta,RVec<float> jet_phi,RVec<float> jet_m){

    TLorentzVector tot,temp;
    tot.SetPtEtaPhiM(photon_pt[0],photon_eta[0],photon_phi[0],0.0);

    for(unsigned int idx = 0; idx < jet_pt.size(); idx++){
      temp.SetPtEtaPhiM(jet_pt[idx],jet_eta[idx],jet_phi[idx],jet_m[idx]);
      tot = tot + temp;
    }

    for(unsigned int idx = 0; idx < el_pt.size(); idx++){
      temp.SetPtEtaPhiM(el_pt[idx],el_eta[idx],el_phi[idx],0.000511);
      tot = tot + temp;
    }

    for(unsigned int idx = 0; idx < mu_pt.size(); idx++){
      temp.SetPtEtaPhiM(mu_pt[idx],mu_eta[idx],mu_phi[idx],0.1057);
      tot = tot + temp;
    }

    return tot.Et();

}

float get_st(RVec<float> photon_pt,RVec<bool> photon_sig,
          RVec<float> el_pt,RVec<float> el_sig,
          RVec<float> mu_pt,RVec<float> mu_sig,
          RVec<float> jet_pt,RVec<bool> jet_isgood){

    float result = 0;

    for(unsigned int idx=0; idx < photon_pt.size(); ++idx) {
      if (!photon_sig[idx]) continue;
      result += photon_pt[idx];
    }

    for(unsigned int idx = 0; idx < jet_pt.size(); idx++){
      if (!jet_isgood[idx]) continue;
      result += jet_pt[idx];
    }

    for(unsigned int idx = 0; idx < el_pt.size(); idx++){
      if (!el_sig[idx]) continue;
      result += el_pt[idx];
    }

    for(unsigned int idx = 0; idx < mu_pt.size(); idx++){
      if (!mu_sig[idx]) continue;
      result += mu_pt[idx];
    }

    return result;
}



float get_weight(float w_lumi ,float w_year, RVec<float> llphoton_l1_masserr,RVec<float> llphoton_l2_masserr,RVec<float> llphoton_ph_masserr, bool isNotSig){
    float dm = 1.0;
    if(isNotSig){dm=1;} else {
      float dml1,dml2,dmph;
      dml1 = llphoton_l1_masserr[0];
      dml2 = llphoton_l2_masserr[0];
      dmph = llphoton_ph_masserr[0];
      dm = sqrt(dml1 * dml1 + dml2 * dml2 + dmph * dmph);
    }

    //return weight;
    //if(SampleType == 2016) { return w_lumi*36.32264/dm;}
    //if(SampleType == 2017) { return w_lumi*41.52756/dm;}
    //if(SampleType == 2018) { return w_lumi*59.67377/dm;}
    //return weight/dm;
    return w_year*w_lumi;
}


bool get_trigger(RVec<int> ll_lepid,
                 int nel,RVec<float> el_sig,RVec<float> el_pt,
                 bool trigs_2016_e,bool trigs_2016_ee,
                 bool trigs_2017_e,bool trigs_2017_ee,
                 bool trigs_2018_e,bool trigs_2018_ee,
                 int nmu,RVec<float> mu_sig,RVec<float> mu_pt,
                 bool trigs_2016_mu,bool trigs_2016_mumu,
                 bool trigs_2017_mu,bool trigs_2017_mumu,
                 bool trigs_2018_mu,bool trigs_2018_mumu,
                 int year
                 ) {
  float signal_lead_electron_pt = -999;
  float signal_sublead_electron_pt = -999;
  for (unsigned iPart = 0; iPart<el_pt.size(); iPart++) {
    if (el_sig.at(iPart)<1) continue;
    if (signal_lead_electron_pt < -998) signal_lead_electron_pt = el_pt.at(iPart);
    else if (signal_sublead_electron_pt < -998) {
      signal_sublead_electron_pt = el_pt.at(iPart);
      break;
    }
  }
  bool trigs_pT_2016_el = (trigs_2016_e && signal_lead_electron_pt > 30) || (trigs_2016_ee && signal_lead_electron_pt > 25 && signal_sublead_electron_pt > 15);
  bool trigs_pT_2017_el = (trigs_2017_e && signal_lead_electron_pt > 35) || (trigs_2017_ee && signal_lead_electron_pt > 25 && signal_sublead_electron_pt > 15); 
  bool trigs_pT_2018_el = (trigs_2018_e && signal_lead_electron_pt > 35) || (trigs_2018_ee && signal_lead_electron_pt > 25 && signal_sublead_electron_pt > 15);
  bool trigs_el_pT = false;
  if (year == 2016) trigs_el_pT = trigs_pT_2016_el;
  else if (year == 2017) trigs_el_pT = trigs_pT_2017_el;
  else if (year == 2018) trigs_el_pT = trigs_pT_2018_el;

  float signal_lead_muon_pt = -999;
  float signal_sublead_muon_pt = -999;
  for (unsigned iPart = 0; iPart<mu_pt.size(); iPart++) {
    if (mu_sig.at(iPart)<1) continue;
    if (signal_lead_muon_pt < -998) signal_lead_muon_pt = mu_pt.at(iPart);
    else if (signal_sublead_muon_pt < -998) {
      signal_sublead_muon_pt = mu_pt.at(iPart);
      break;
    }
  }
  bool trigs_pT_2016_mu = (trigs_2016_mu && signal_lead_muon_pt > 25) || (trigs_2016_mumu && signal_lead_muon_pt > 20 && signal_sublead_muon_pt > 10);
  bool trigs_pT_2017_mu = (trigs_2017_mu && signal_lead_muon_pt > 28) || (trigs_2017_mumu && signal_lead_muon_pt > 20 && signal_sublead_muon_pt > 10);
  bool trigs_pT_2018_mu = (trigs_2018_mu && signal_lead_muon_pt > 25) || (trigs_2018_mumu && signal_lead_muon_pt > 20 && signal_sublead_muon_pt > 10);
  bool trigs_mu_pT = false;
  if (year == 2016) trigs_mu_pT = trigs_pT_2016_mu;
  else if (year == 2017) trigs_mu_pT = trigs_pT_2017_mu;
  else if (year == 2018) trigs_mu_pT = trigs_pT_2018_mu;

  return ((trigs_mu_pT&&nmu>=2&&ll_lepid.at(0)==13) || (trigs_el_pT&&nel>=2&&ll_lepid.at(0)==11));
}



float get_luminosity(unsigned int slot, const ROOT::RDF::RSampleInfo &id) {
  float luminosity = 0;
  if (id.Contains("2016APV/mc")) luminosity = 19.51;
  else if (id.Contains("2016/mc"))  luminosity = 16.80;
  else if (id.Contains("2017/mc")) luminosity = 41.48;
  else if (id.Contains("2018/mc")) luminosity = 59.83;
  //cout<<id.AsString()<<" "<<luminosity<<endl;
  return luminosity;
}

int get_year(unsigned int slot, const ROOT::RDF::RSampleInfo &id) {
  int year = 0;
  if (id.Contains("2016APV/mc")) year = 2016;
  else if (id.Contains("2016/mc"))  year = 2016;
  else if (id.Contains("2017/mc")) year = 2017;
  else if (id.Contains("2018/mc")) year = 2018;
  return year;
}

float get_j1_pt(int njet, RVec<int> jet_isgood, RVec<float> jet_pt){
  if (njet<1) return -999;
  for (unsigned iPart = 0; iPart<jet_pt.size(); iPart++) {
      if (jet_isgood.at(iPart)) return jet_pt.at(iPart);
  }
  return -999;
}
float get_j1_eta(int njet, RVec<int> jet_isgood, RVec<float> jet_eta){
  if (njet<1) return -999;
  for (unsigned iPart = 0; iPart<jet_eta.size(); iPart++) {
      if (jet_isgood.at(iPart)) return jet_eta.at(iPart);
  }
  return -999;
}
float get_j1_phi(int njet, RVec<int> jet_isgood, RVec<float> jet_phi){
  if (njet<1) return -999;
  for (unsigned iPart = 0; iPart<jet_phi.size(); iPart++) {
      if (jet_isgood.at(iPart)) return jet_phi.at(iPart);
  }
  return -999;
}
float get_llyj_dphi(float llphoton_phi, int njet, RVec<int> jet_isgood, RVec<float> jet_phi){
  if (njet<1) return -999;
  float signal_jet_phi = -999;
  for (unsigned iPart = 0; iPart<jet_phi.size(); iPart++) {
      if (jet_isgood.at(iPart)) {
        signal_jet_phi = jet_phi.at(iPart);
        break;
      }
  }
  if (signal_jet_phi<-998) return signal_jet_phi;
  float dphi_lead = TVector2::Phi_mpi_pi(llphoton_phi - signal_jet_phi);
  return dphi_lead;
}
float get_yj1_deta(float photon_eta, int njet, RVec<int> jet_isgood, RVec<float> jet_eta){
  if (njet<1) return -999;
  float signal_jet_eta = -999;
  for (unsigned iPart = 0; iPart<jet_eta.size(); iPart++) {
      if (jet_isgood.at(iPart)) {
        signal_jet_eta = jet_eta.at(iPart);
        break;
      }
  }
  if (signal_jet_eta<-998) return signal_jet_eta;
  float deta_lead = photon_eta - signal_jet_eta;
  return deta_lead;
}
float get_yj1_dr(float photon_eta, float photon_phi, int njet, RVec<int> jet_isgood, RVec<float> jet_eta, RVec<float> jet_phi){
  if (njet<1) return -999;
  float signal_jet_eta = -999;
  float signal_jet_phi = -999;
  for (unsigned iPart = 0; iPart<jet_eta.size(); iPart++) {
      if (jet_isgood.at(iPart)) {
        signal_jet_eta = jet_eta.at(iPart);
        signal_jet_phi = jet_phi.at(iPart);
        break;
      }
  }
  if (signal_jet_eta<-998) return signal_jet_eta;
  return get_dr(photon_eta, photon_phi, signal_jet_eta, signal_jet_phi);
}
float get_llyj1_ptbal(float ll_pt, float ll_eta, float ll_phi, float photon_pt, float photon_eta, float photon_phi, int njet, RVec<int> jet_isgood, RVec<float> jet_pt, RVec<float> jet_eta, RVec<float> jet_phi){
  if (njet<1) return -999;
  float signal_jet_eta = -999;
  float signal_jet_phi = -999;
  float signal_jet_pt = -999;
  for (unsigned iPart = 0; iPart<jet_eta.size(); iPart++) {
      if (jet_isgood.at(iPart)) {
        signal_jet_pt = jet_pt.at(iPart);
        signal_jet_eta = jet_eta.at(iPart);
        signal_jet_phi = jet_phi.at(iPart);
        break;
      }
  }
  if (signal_jet_eta<-998) return signal_jet_eta;
  TVector3 zboson; zboson.SetPtEtaPhi(ll_pt, ll_eta, ll_phi);
  TVector3 gamma; gamma.SetPtEtaPhi(photon_pt, photon_eta, photon_phi);
  TVector3 jet; jet.SetPtEtaPhi(signal_jet_pt, signal_jet_eta, signal_jet_phi);
  return (zboson+gamma+jet).Pt()/(zboson.Pt()+gamma.Pt()+jet.Pt());
}


float get_j2_pt(int njet, RVec<int> jet_isgood, RVec<float> jet_pt){
  if (njet<2) return -999;
  bool sublead = false;
  for (unsigned iPart = 0; iPart<jet_pt.size(); iPart++) {
    if (jet_isgood.at(iPart) == 0) continue;
    if (sublead == true) return jet_pt.at(iPart);
    sublead = true;
  }
  return -999;
}

float get_llg_ptt(RVec<float> photon_pt, RVec<float> photon_eta, RVec<float> photon_phi, 
                  RVec<float> llphoton_pt, RVec<float> llphoton_eta, RVec<float> llphoton_phi,
                  RVec<float> ll_pt, RVec<float> ll_eta, RVec<float> ll_phi) {
  TVector3 gamma; gamma.SetPtEtaPhi(photon_pt[0], photon_eta[0], photon_phi[0]);
  TVector3 higgs; higgs.SetPtEtaPhi(llphoton_pt[0], llphoton_eta[0], llphoton_phi[0]);
  TVector3 zboson; zboson.SetPtEtaPhi(ll_pt[0], ll_eta[0], ll_phi[0]);
  gamma.SetZ(0); higgs.SetZ(0); zboson.SetZ(0);
  return higgs.Cross((zboson-gamma).Unit()).Mag();
}

float get_flavor(RVec<int> ll_lepid) {
  if (ll_lepid[0] == 11) return 1.;
  if (ll_lepid[0] == 13) return 2.;
  return 0.;
}

float get_tru_leplep_m(RVec<float> mc_id, RVec<float> mc_status, RVec<float> mc_mom, RVec<float> mc_pt, RVec<float> mc_eta, RVec<float> mc_phi, RVec<float> mc_mass) {
  float leplep_m = -999;
  TLorentzVector lep_plus;
  TLorentzVector lep_minus;
  bool set_lep_plus = false;
  bool set_lep_minus = false;
  for (unsigned imc = 0; imc < mc_id.size(); ++imc) {
    if (mc_id[imc]==23) {
      leplep_m = mc_mass[imc];
      break;
    }
    if (mc_status[imc]==23)  {
      if (mc_id[imc]==11 || mc_id[imc]==13) {
        lep_plus.SetPtEtaPhiM(mc_pt[imc],mc_eta[imc],mc_phi[imc],mc_mass[imc]);
        set_lep_plus = true;
      } else if (mc_id[imc]==-11 || mc_id[imc]==-13) {
        lep_minus.SetPtEtaPhiM(mc_pt[imc],mc_eta[imc],mc_phi[imc],mc_mass[imc]);
        set_lep_minus = true;
      }
    }
    if (set_lep_plus && set_lep_minus) break;
  }
  if (leplep_m>-998) return leplep_m;
  if (set_lep_minus && set_lep_plus) return (lep_plus+lep_minus).M();
  return -999;
}

""")

if __name__=='__main__':
  #ROOT.EnableImplicitMT() # Turn off to keep ntuple event sequence
  start_time = time.time()
 
  years = ["2016APV","2016","2017","2018"]
  #years = ["2016"]

  #make n-tuples
  defines = [
       # BDT variables
       ('y_mva','photon_idmva[0]'),
       ('yl_drmin','photon_drmin[0]'),
       ('yl_drmax','get_max_dr(photon_eta,photon_phi,el_eta,el_phi,mu_eta,mu_phi,ll_lepid,ll_i1,ll_i2)'),
       ('ht','H_t(photon_pt,photon_eta,photon_phi,el_pt,el_eta,el_phi,mu_pt,mu_eta,mu_phi,ll_lepid,ll_i1,ll_i2,jet_pt,jet_eta,jet_phi,jet_m)'),
       ('st','get_st(photon_pt,photon_sig,el_pt,el_sig,mu_pt,mu_sig,jet_pt,jet_isgood)'),
       ('lly_ptmass','llphoton_pt[0]/llphoton_m[0]'),
       ('cosTheta','llphoton_cosTheta[0]'),
       ('costheta','llphoton_costheta[0]'),
       ('phi','llphoton_psi[0]'),
       ('y_res','photon_energyErr[0]/photon_pt[0]'),
       ('y_eta','photon_eta[0]'),
       ('y_pt','photon_pt[0]'),     
       ('y_ptmass','photon_pt[0]/llphoton_m[0]'),      
       ('l1_eta','get_l1_eta(el_pt,el_eta,mu_pt,mu_eta,ll_lepid,ll_i1,ll_i2)'),
       ('l2_eta','get_l2_eta(el_pt,el_eta,mu_pt,mu_eta,ll_lepid,ll_i1,ll_i2)'),
       ('yl_drminmass','photon_drmin[0]/llphoton_m[0]'),

       # event filters
       ('trigger','get_trigger(ll_lepid,nel,el_sig,el_pt,HLT_Ele27_WPTight_Gsf,HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ,HLT_Ele32_WPTight_Gsf_L1DoubleEG,HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL,HLT_Ele32_WPTight_Gsf,HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL,nmu,mu_sig,mu_pt,HLT_IsoMu24 || HLT_IsoTkMu24,HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL || HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ || HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL || HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ,HLT_IsoMu27,HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8 || HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8,HLT_IsoMu24,HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8, year)'),
       ('pass_filter', 'pass_goodv&&pass_cschalo_tight&&pass_hbhe&&pass_hbheiso&&pass_ecaldeadcell&&pass_badpfmu&&pass_badpfmudz&&pass_hfnoisyhits&&pass_eebadsc&&pass_badcalib'),

       # POI
       ('lly_m','llphoton_m[0]'),
       ('l1_pt','get_l1_pt(el_pt,mu_pt,ll_lepid,ll_i1,ll_i2)'),
       ('l2_pt','get_l2_pt(el_pt,mu_pt,ll_lepid,ll_i1,ll_i2)'),
       ('l1_phi','get_l1_phi(el_pt,el_phi,mu_pt,mu_phi,ll_lepid,ll_i1,ll_i2)'),
       ('l2_phi','get_l2_phi(el_pt,el_phi,mu_pt,mu_phi,ll_lepid,ll_i1,ll_i2)'),
       ('y_phi','photon_phi[0]'),     
       ('y_id80', 'photon_id80[0]'),
       ('leplep_pt', 'll_pt[0]'),
       ('leplep_eta', 'll_eta[0]'),
       ('leplep_phi', 'll_phi[0]'),
       ('leplep_m', 'll_m[0]'),
       ('leplep_flavor', 'get_flavor(ll_lepid)'),
       ('leplep_charge', 'll_charge[0]'),
       ('lly_pt','llphoton_pt[0]'),
       ('lly_eta','llphoton_eta[0]'),
       ('lly_phi','llphoton_phi[0]'),
       ('lly_ptt', 'get_llg_ptt(photon_pt, photon_eta, photon_phi, llphoton_pt, llphoton_eta, llphoton_phi, ll_pt, ll_eta, ll_phi)'),
       ('tru_leplep_m', 'get_tru_leplep_m(mc_id,mc_status,mc_mom,mc_pt,mc_eta,mc_phi,mc_mass)'),
       ('event_number','event'),
       ('j1_pt','get_j1_pt(njet,jet_isgood, jet_pt)'),
       ('j1_eta','get_j1_eta(njet,jet_isgood, jet_eta)'),
       ('j1_phi','get_j1_phi(njet,jet_isgood, jet_phi)'),
       ('llyj_dphi', 'get_llyj_dphi(llphoton_phi[0], njet, jet_isgood, jet_phi)'),
       ('yj1_deta', 'get_yj1_deta(photon_eta[0], njet, jet_isgood, jet_eta)'),
       ('yj1_dr', 'get_yj1_dr(photon_eta[0], photon_phi[0], njet, jet_isgood, jet_eta, jet_phi)'),
       ('llyj1_ptbal', 'get_llyj1_ptbal(ll_pt[0], ll_eta[0], ll_phi[0], photon_pt[0], photon_eta[0], photon_phi[0], njet, jet_isgood, jet_pt, jet_eta, jet_phi)'),
       ('lly_res', 'sqrt(llphoton_l1_masserr[0]*llphoton_l1_masserr[0]+llphoton_l2_masserr[0]*llphoton_l2_masserr[0]+llphoton_ph_masserr[0]*llphoton_ph_masserr[0])'),
       ]
 
  # Select only the needed branches to reduce space
  branches = ['y_mva','yl_drmin','yl_drmax','lly_ptmass','cosTheta','costheta','ht', 'st','phi','y_res','y_eta','y_pt','y_ptmass','l1_eta','l2_eta', 'yl_drminmass']
  branches.extend(['lly_m','l1_pt','l2_pt','l1_phi','l2_phi','y_phi','y_id80'])
  branches.extend(['leplep_pt','leplep_eta','leplep_phi','leplep_m','leplep_flavor'])
  branches.extend(['lly_pt','lly_eta','lly_phi','lly_ptt'])
  branches.extend(['met','nlep', 'nllphoton', 'npv', 'lly_res'])
  branches.extend(['j1_pt','j1_eta','j1_phi', 'njet', 'llyj_dphi', 'yj1_deta', 'yj1_dr', 'llyj1_ptbal'])
  branches.extend(['tru_leplep_m'])
  branches.extend(['year', 'luminosity', 'w_lumiXyear', 'weightXyear', 'type', 'pass_filter','event_number', 'trigger', 'use_event'])

  #make n-tuples
  cuts = ['trigger', 'pass_filter', 'nllphoton>=1',
          'use_event']

  names = 'ggf_ntuples'
  base_dir  = '/net/cms11/cms11r0/pico/NanoAODv9/htozgamma_kingscanyon_v1/'
  pico_type = '/mc/merged_zgmc_llg/'
  sig_samples = ['*GluGluHToZG_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8*.root','*VBFHToZG_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8*.root']
  bkg_samples = ['*ZGToLLG_01J_5f_lowMLL_lowGPt_TuneCP5_13TeV-amcatnloFXFX-pythia8*.root','*DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8*.root','*ZGamma2JToGamma2L2J_EWK_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8*.root']
  print([base_dir + year + pico_type + sig for sig in sig_samples for year in years])
  print([base_dir + year + pico_type  + bkg for bkg in bkg_samples for year in years])

  #Make signal ntuples
  for sig in sig_samples:
    dataset_name=sig.replace('*','').replace('.root','')
    write_ntuples([base_dir + year + pico_type + sig for year in years],  cuts,  
             (names + '_' + dataset_name + '_' + ''.join(years) + '.root'),  defines,  'tree',  branches)

  #Make bkg ntuples               
  for bkg in bkg_samples:
    dataset_name=bkg.replace('*','').replace('.root','')
    write_ntuples([base_dir + year + pico_type  + bkg for year in years],  cuts,
             (names + '_' + dataset_name + '_' + ''.join(years) + '.root'),  defines,  'tree',  branches)

  elapsed_time = time.time() - start_time
  print(f'Elapsed time: {elapsed_time}')
