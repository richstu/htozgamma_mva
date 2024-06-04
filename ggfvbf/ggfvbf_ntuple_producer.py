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
RVec<int> truth_matched(RVec<float> mc_pt, RVec<float> mc_eta, RVec<float> mc_phi,RVec<float> mc_m, RVec<float> mc_id, RVec<float> mc_momidx, 
                        RVec<float> jet_pt, RVec<float> jet_eta, RVec<float> jet_phi, RVec<float> jet_m,RVec<bool> jet_isgood){
   RVec<int> jet_idx = {-1,-1}; 
   TLorentzVector jet,mc;

   for(int idx = 0; idx < jet_pt.size(); idx++){
        if(!jet_isgood[idx]){continue;}

        //IF YOU DONT WANT TRUTH-MATCHED JETS
        if(jet_idx[0]==-1){jet_idx[0]=idx;continue;}
          jet_idx[1]=idx;break;
          
       
        jet.SetPtEtaPhiM(jet_pt[idx],jet_eta[idx],jet_phi[idx],jet_m[idx]);

        for(int mc_idx = 0; mc_idx < mc_pt.size(); mc_idx++){
          if( (fabs(mc_id[mc_idx]) >10) ||(mc_momidx[mc_idx]==-1)||( mc_id[ mc_momidx[mc_idx] ] == 21) ){ continue; }
          mc.SetPtEtaPhiM(mc_pt[mc_idx],mc_eta[mc_idx],mc_phi[mc_idx],mc_m[mc_idx]);

          if(mc.DeltaR(jet) > 0.1 || fabs(mc.Pt() - jet.Pt()) > 0.1*jet.Pt()){ continue; }

          if(jet_idx[0]==-1){
            jet_idx[0] = idx;
            break;
          }
          jet_idx[1] = idx;
          break;

        } 

     if(jet_idx[1] > -1){break;}
    
   }
    
   return jet_idx;
}

bool truth_matched_bool(RVec<float> mc_pt, RVec<float> mc_eta, RVec<float> mc_phi,RVec<float> mc_m, RVec<float> mc_id, RVec<float> mc_momidx, 
                        RVec<float> jet_pt, RVec<float> jet_eta, RVec<float> jet_phi, RVec<float> jet_m,RVec<bool> jet_isgood){
    RVec<int> jet_idx = truth_matched(mc_pt,mc_eta,mc_phi,mc_m,mc_id,mc_momidx,jet_pt,jet_eta,jet_phi,jet_m,jet_isgood);
    if(jet_idx[0] == -1 || jet_idx[1] == -1){ return false; }
    return true;
}

double deltaR(double p1_eta, double p1_phi, double p2_eta, double p2_phi) {
  Double_t deta = p1_eta-p2_eta;
  Double_t dphi = TVector2::Phi_mpi_pi(p1_phi-p2_phi);
  return TMath::Sqrt( deta*deta+dphi*dphi );
}

float get_max_dr(RVec<float> photon_eta, RVec<float> photon_phi, 
    RVec<float> el_eta, RVec<float> el_phi, RVec<float> mu_eta,
    RVec<float> mu_phi, RVec<int> ll_lepid, RVec<int> ll_i1,
    RVec<int> ll_i2) {
  float dr1, dr2;
  if (ll_lepid[0]==11) {
    dr1 = deltaR(photon_eta[0],photon_phi[0],el_eta[ll_i1[0]],el_phi[ll_i1[0]]);
    dr2 = deltaR(photon_eta[0],photon_phi[0],el_eta[ll_i2[0]],el_phi[ll_i2[0]]);
    return dr1 > dr2 ? dr1 : dr2;
  }
  dr1 = deltaR(photon_eta[0],photon_phi[0],mu_eta[ll_i1[0]],mu_phi[ll_i1[0]]);
  dr2 = deltaR(photon_eta[0],photon_phi[0],mu_eta[ll_i2[0]],mu_phi[ll_i2[0]]);
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

float get_dphi(float phi1, float phi2){
  const double PI = 3.1415;
  double dphi = fmod(fabs(phi2-phi1), 2.*PI);
    dphi = dphi>PI ? 2.*PI-dphi : dphi;

  return dphi;
}

float jj_dphi(RVec<float> jet_phi){ return get_dphi(jet_phi[0],jet_phi[1]); }

float jj_dphi(RVec<float> mc_pt, RVec<float> mc_eta, RVec<float> mc_phi,RVec<float> mc_m, RVec<float> mc_id, RVec<float> mc_momidx, 
              RVec<float> jet_pt, RVec<float> jet_eta, RVec<float> jet_phi, RVec<float> jet_m, RVec<bool> jet_isgood){ 
    RVec<int> jet_idx = truth_matched(mc_pt,mc_eta,mc_phi,mc_m,mc_id,mc_momidx,jet_pt,jet_eta,jet_phi,jet_m,jet_isgood);
    return get_dphi(jet_phi[jet_idx[0]],jet_phi[jet_idx[1]]);
}

float get_Zy_dphi(RVec<float> ll_phi, RVec<float> photon_phi){ return get_dphi(ll_phi[0],photon_phi[0]);  }

float get_Zyjj_dphi(RVec<float> ll_pt,RVec<float> ll_eta,RVec<float> ll_phi,RVec<float> ll_m, 
                RVec<float> photon_pt, RVec<float> photon_eta, RVec<float> photon_phi,
                float dijet_pt, float dijet_eta, float dijet_phi, float dijet_m){

  TLorentzVector y,Z,jj;
  y.SetPtEtaPhiM(photon_pt[0],photon_eta[0],photon_phi[0],0);
  Z.SetPtEtaPhiM(ll_pt[0],ll_eta[0],ll_phi[0],ll_m[0]);
  jj.SetPtEtaPhiM(dijet_pt,dijet_eta,dijet_phi,dijet_m);
  return get_dphi((Z+y).Phi(),jj.Phi());  
               
}

float get_Zyjj_dphi(RVec<float> mc_pt, RVec<float> mc_eta, RVec<float> mc_phi,RVec<float> mc_m, RVec<float> mc_id, RVec<float> mc_momidx,
                    RVec<float> ll_pt,RVec<float> ll_eta,RVec<float> ll_phi,RVec<float> ll_m, 
                    RVec<float> photon_pt, RVec<float> photon_eta, RVec<float> photon_phi,
                    RVec<float> jet_pt, RVec<float> jet_eta, RVec<float> jet_phi, RVec<float> jet_m, RVec<bool> jet_isgood){

  RVec<int> jet_idx = truth_matched(mc_pt,mc_eta,mc_phi,mc_m,mc_id,mc_momidx,jet_pt,jet_eta,jet_phi,jet_m,jet_isgood);
  TLorentzVector y,Z,jj;
  jj.SetPtEtaPhiM(jet_pt[jet_idx[0]],jet_eta[jet_idx[0]],jet_phi[jet_idx[0]],jet_m[jet_idx[0]]);
  Z.SetPtEtaPhiM(jet_pt[jet_idx[1]],jet_eta[jet_idx[1]],jet_phi[jet_idx[1]],jet_m[jet_idx[1]]);
  jj = Z+jj;

  y.SetPtEtaPhiM(photon_pt[0],photon_eta[0],photon_phi[0],0);
  Z.SetPtEtaPhiM(ll_pt[0],ll_eta[0],ll_phi[0],ll_m[0]);
  return get_dphi((Z+y).Phi(),jj.Phi());  
 
 }

float get_Zyjj_dr(RVec<float> ll_pt,RVec<float> ll_eta,RVec<float> ll_phi,RVec<float> ll_m, 
                RVec<float> photon_pt, RVec<float> photon_eta, RVec<float> photon_phi,
                float dijet_pt, float dijet_eta, float dijet_phi, float dijet_m){

  TLorentzVector y,Z,jj;
  y.SetPtEtaPhiM(photon_pt[0],photon_eta[0],photon_phi[0],0);
  Z.SetPtEtaPhiM(ll_pt[0],ll_eta[0],ll_phi[0],ll_m[0]);
  jj.SetPtEtaPhiM(dijet_pt,dijet_eta,dijet_phi,dijet_m);
  return (Z+y).DeltaR(jj);  
               
}

float get_Zyjj_dr(RVec<float> mc_pt, RVec<float> mc_eta, RVec<float> mc_phi,RVec<float> mc_m, RVec<float> mc_id, RVec<float> mc_momidx,
      RVec<float> ll_pt,RVec<float> ll_eta,RVec<float> ll_phi,RVec<float> ll_m, 
      RVec<float> photon_pt, RVec<float> photon_eta, RVec<float> photon_phi,
      RVec<float> jet_pt, RVec<float> jet_eta, RVec<float> jet_phi, RVec<float> jet_m, RVec<bool> jet_isgood){


      RVec<int> jet_idx = truth_matched(mc_pt,mc_eta,mc_phi,mc_m,mc_id,mc_momidx,jet_pt,jet_eta,jet_phi,jet_m,jet_isgood);
      TLorentzVector y,Z,jj;
      jj.SetPtEtaPhiM(jet_pt[jet_idx[0]],jet_eta[jet_idx[0]],jet_phi[jet_idx[0]],jet_m[jet_idx[0]]);
      Z.SetPtEtaPhiM(jet_pt[jet_idx[1]],jet_eta[jet_idx[1]],jet_phi[jet_idx[1]],jet_m[jet_idx[1]]);
      jj = Z+jj;

      y.SetPtEtaPhiM(photon_pt[0],photon_eta[0],photon_phi[0],0);
      Z.SetPtEtaPhiM(ll_pt[0],ll_eta[0],ll_phi[0],ll_m[0]);
      return (Z+y).DeltaR(jj);  

}

float get_deta(float eta1, float eta2){
  return fabs(eta1-eta2);
}

float jj_deta(RVec<float> jet_eta){ return get_deta(jet_eta[0],jet_eta[1]); }

float jj_deta(RVec<float> mc_pt, RVec<float> mc_eta, RVec<float> mc_phi,RVec<float> mc_m, RVec<float> mc_id, RVec<float> mc_momidx, 
              RVec<float> jet_pt, RVec<float> jet_eta, RVec<float> jet_phi, RVec<float> jet_m, RVec<bool> jet_isgood){ 

    RVec<int> jet_idx = truth_matched(mc_pt,mc_eta,mc_phi,mc_m,mc_id,mc_momidx,jet_pt,jet_eta,jet_phi,jet_m,jet_isgood);
    return get_deta(jet_eta[jet_idx[0]],jet_eta[jet_idx[1]]);
}

float zep_var(RVec<float> ll_pt,RVec<float> ll_eta,RVec<float> ll_phi, RVec<float> ll_m,
              RVec<float> jet_pt,RVec<float> jet_eta,RVec<float> jet_phi, RVec<float> jet_m,
              RVec<float> photon_pt,RVec<float> photon_eta,RVec<float> photon_phi){
 
  TLorentzVector y,Z,j1,j2;
  y.SetPtEtaPhiM(photon_pt[0],photon_eta[0],photon_phi[0],0);
  Z.SetPtEtaPhiM(ll_pt[0],ll_eta[0],ll_phi[0],ll_m[0]);
  j1.SetPtEtaPhiM(jet_pt[0],jet_eta[0],jet_phi[0],jet_m[0]);
  j2.SetPtEtaPhiM(jet_pt[1],jet_eta[1],jet_phi[1],jet_m[1]);
  //return get_deta(photon_eta[0],0.5*(get_deta(jet_eta[0],-1.0*jet_eta[1])) );
  return fabs(((y+Z).Rapidity() - 0.5*(j1.Rapidity() + j2.Rapidity()))/fabs(j1.Rapidity() - j2.Rapidity() ));

}

float zep_var(RVec<float> mc_pt, RVec<float> mc_eta, RVec<float> mc_phi,RVec<float> mc_m, RVec<float> mc_id, RVec<float> mc_momidx, 
              RVec<float> ll_pt,RVec<float> ll_eta,RVec<float> ll_phi, RVec<float> ll_m,
              RVec<float> jet_pt,RVec<float> jet_eta,RVec<float> jet_phi, RVec<float> jet_m, RVec<bool> jet_isgood,
              RVec<float> photon_pt,RVec<float> photon_eta,RVec<float> photon_phi){

  RVec<int> jet_idx = truth_matched(mc_pt,mc_eta,mc_phi,mc_m,mc_id,mc_momidx,jet_pt,jet_eta,jet_phi,jet_m,jet_isgood);
  TLorentzVector y,Z,j1,j2;
  y.SetPtEtaPhiM(photon_pt[0],photon_eta[0],photon_phi[0],0);
  Z.SetPtEtaPhiM(ll_pt[0],ll_eta[0],ll_phi[0],ll_m[0]);
  j1.SetPtEtaPhiM(jet_pt[jet_idx[0]],jet_eta[jet_idx[0]],jet_phi[jet_idx[0]],jet_m[jet_idx[0]]);
  j2.SetPtEtaPhiM(jet_pt[jet_idx[1]],jet_eta[jet_idx[1]],jet_phi[jet_idx[1]],jet_m[jet_idx[1]]);
  //return get_deta(photon_eta[0],0.5*(get_deta(jet_eta[0],-1.0*jet_eta[1])) );
  return fabs(((y+Z).Rapidity() - 0.5*(j1.Rapidity() + j2.Rapidity()))/fabs(j1.Rapidity() - j2.Rapidity() ));

}

float drmax_yj(RVec<float> photon_eta, RVec<float> photon_phi,RVec<float> jet_eta,RVec<float> jet_phi){
    float dr1,dr2;
    dr1 = deltaR(photon_eta[0],photon_phi[0],jet_eta[0],jet_phi[0]);
    dr2 = deltaR(photon_eta[0],photon_phi[0],jet_eta[1],jet_phi[0]);
    return dr1 > dr2 ? dr1 : dr2;
}

float drmax_yj(RVec<float> mc_pt, RVec<float> mc_eta, RVec<float> mc_phi,RVec<float> mc_m, RVec<float> mc_id, RVec<float> mc_momidx, 
              RVec<float> jet_pt, RVec<float> jet_eta, RVec<float> jet_phi, RVec<float> jet_m, RVec<bool> jet_isgood,RVec<float> photon_eta, RVec<float> photon_phi){ 
  RVec<int> jet_idx = truth_matched(mc_pt,mc_eta,mc_phi,mc_m,mc_id,mc_momidx,jet_pt,jet_eta,jet_phi,jet_m,jet_isgood);
  float dr1,dr2;
  dr1 = deltaR(photon_eta[0],photon_phi[0],jet_eta[jet_idx[0]],jet_phi[jet_idx[0]]);
  dr2 = deltaR(photon_eta[0],photon_phi[0],jet_eta[jet_idx[1]],jet_phi[jet_idx[1]]);
  return dr1 > dr2 ? dr1 : dr2;
}

float drmin_yj(RVec<float> photon_eta, RVec<float> photon_phi,RVec<float> jet_eta, RVec<float> jet_phi){
    float dr1,dr2;
    dr1 = deltaR(photon_eta[0],photon_phi[0],jet_eta[0],jet_phi[0]);
    dr2 = deltaR(photon_eta[0],photon_phi[0],jet_eta[1],jet_phi[1]);
    return dr1 < dr2 ? dr1 : dr2;
}

float drmin_yj(RVec<float> mc_pt, RVec<float> mc_eta, RVec<float> mc_phi,RVec<float> mc_m, RVec<float> mc_id, RVec<float> mc_momidx, 
              RVec<float> jet_pt, RVec<float> jet_eta, RVec<float> jet_phi, RVec<float> jet_m, RVec<bool> jet_isgood,RVec<float> photon_eta, RVec<float> photon_phi){ 
  RVec<int> jet_idx = truth_matched(mc_pt,mc_eta,mc_phi,mc_m,mc_id,mc_momidx,jet_pt,jet_eta,jet_phi,jet_m,jet_isgood);
  float dr1,dr2;
  dr1 = deltaR(photon_eta[0],photon_phi[0],jet_eta[jet_idx[0]],jet_phi[jet_idx[0]]);
  dr2 = deltaR(photon_eta[0],photon_phi[0],jet_eta[jet_idx[1]],jet_phi[jet_idx[1]]);
  return dr1 < dr2 ? dr1 : dr2;
}

float drmin_l1j(RVec<int> ll_lepid, RVec<float> el_eta, RVec<float> el_phi,RVec<float> mu_eta, RVec<float> mu_phi, RVec<float> jet_eta, RVec<float> jet_phi){
    float dr1,dr2;
    if(ll_lepid[0]==11){ 
      dr1 = deltaR(el_eta[0],el_phi[0],jet_eta[0],jet_phi[0]); 
      dr2 = deltaR(el_eta[0],el_phi[0],jet_eta[1],jet_phi[1]);
    } else {
      dr1 = deltaR(mu_eta[0],mu_phi[0],jet_eta[0],jet_phi[0]); 
      dr2 = deltaR(mu_eta[0],mu_phi[0],jet_eta[1],jet_phi[1]);
    }
    return dr1 < dr2 ? dr1 : dr2;
}

float drmin_l2j(RVec<int> ll_lepid, RVec<float> el_eta, RVec<float> el_phi,RVec<float> mu_eta, RVec<float> mu_phi, RVec<float> jet_eta, RVec<float> jet_phi){
    float dr1,dr2;
    if(ll_lepid[0]==11){ 
      dr1 = deltaR(el_eta[1],el_phi[1],jet_eta[0],jet_phi[0]); 
      dr2 = deltaR(el_eta[1],el_phi[1],jet_eta[1],jet_phi[1]);
    } else {
      dr1 = deltaR(mu_eta[1],mu_phi[1],jet_eta[0],jet_phi[0]); 
      dr2 = deltaR(mu_eta[1],mu_phi[1],jet_eta[1],jet_phi[1]);
    }
    
    return dr1 < dr2 ? dr1 : dr2;
}

float dr_jj(RVec<float> jet_eta, RVec<float> jet_phi){
    float dr1;
    return deltaR(jet_eta[0],jet_phi[0],jet_eta[1],jet_phi[1]);
}

float dr_jj(RVec<float> mc_pt, RVec<float> mc_eta, RVec<float> mc_phi,RVec<float> mc_m, RVec<float> mc_id, RVec<float> mc_momidx, 
            RVec<float> jet_pt, RVec<float> jet_eta, RVec<float> jet_phi, RVec<float> jet_m, RVec<bool> jet_isgood){ 
    RVec<int> jet_idx = truth_matched(mc_pt,mc_eta,mc_phi,mc_m,mc_id,mc_momidx,jet_pt,jet_eta,jet_phi,jet_m,jet_isgood);
    return deltaR(jet_eta[jet_idx[0]],jet_phi[jet_idx[0]],jet_eta[jet_idx[1]],jet_phi[jet_idx[1]]);
}

float pt_bal_func(RVec<float> ll_pt, RVec<float> ll_eta, RVec<float> ll_phi,RVec<float> ll_m, 
               RVec<float> jet_pt,RVec<float> jet_eta, RVec<float> jet_phi, RVec<float> jet_m, 
               RVec<float> photon_pt, RVec<float> photon_eta, RVec<float> photon_phi){

  TLorentzVector Z,y,j1,j2;
  Z.SetPtEtaPhiM(ll_pt[0],ll_eta[0],ll_phi[0],ll_m[0]);
  y.SetPtEtaPhiM(photon_pt[0],photon_eta[0],photon_phi[0],0.0);
  j1.SetPtEtaPhiM(jet_pt[0],jet_eta[0],jet_phi[0],jet_m[0]);
  j2.SetPtEtaPhiM(jet_pt[1],jet_eta[1],jet_phi[1],jet_m[1]);
  return ( (Z+y+j1+j2).Pt()/(Z.Pt() + y.Pt() + j1.Pt() + j2.Pt()) ); 
  }

float pt_bal_func(RVec<float> mc_pt, RVec<float> mc_eta, RVec<float> mc_phi,RVec<float> mc_m, RVec<float> mc_id, RVec<float> mc_momidx, 
                  RVec<float> ll_pt, RVec<float> ll_eta, RVec<float> ll_phi,RVec<float> ll_m, 
                  RVec<float> jet_pt,RVec<float> jet_eta, RVec<float> jet_phi, RVec<float> jet_m, RVec<bool> jet_isgood, 
                  RVec<float> photon_pt, RVec<float> photon_eta, RVec<float> photon_phi){
    RVec<int> jet_idx = truth_matched(mc_pt,mc_eta,mc_phi,mc_m,mc_id,mc_momidx,jet_pt,jet_eta,jet_phi,jet_m,jet_isgood);
    TLorentzVector Z,y,j1,j2;
    Z.SetPtEtaPhiM(ll_pt[0],ll_eta[0],ll_phi[0],ll_m[0]);
    y.SetPtEtaPhiM(photon_pt[0],photon_eta[0],photon_phi[0],0.0);
    j1.SetPtEtaPhiM(jet_pt[jet_idx[0]],jet_eta[jet_idx[0]],jet_phi[jet_idx[0]],jet_m[jet_idx[0]]);
    j2.SetPtEtaPhiM(jet_pt[jet_idx[1]],jet_eta[jet_idx[1]],jet_phi[jet_idx[1]],jet_m[jet_idx[1]]);
    return ( (Z+y+j1+j2).Pt()/(Z.Pt() + y.Pt() + j1.Pt() + j2.Pt()) ); 
}

float pTt_funcVec(RVec<float> ll_pt, RVec<float> ll_eta, RVec<float> ll_phi,RVec<float> ll_m, RVec<float> photon_pt, RVec<float> photon_eta, RVec<float> photon_phi){
    TLorentzVector Z,y;
    Z.SetPtEtaPhiM(ll_pt[0],ll_eta[0],ll_phi[0],ll_m[0]);
    y.SetPtEtaPhiM(photon_pt[0],photon_eta[0],photon_phi[0],0.0);
    return fabs( (Z+y).Vect().Cross( (Z-y).Vect() ).Pz()/( (Z+y).Pt() ) );
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
float get_j1_m(int njet, RVec<int> jet_isgood, RVec<float> jet_m){
  if (njet<1) return -999;
  for (unsigned iPart = 0; iPart<jet_m.size(); iPart++) {
      if (jet_isgood.at(iPart)) return jet_m.at(iPart);
  }
  return -999;
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
float get_j2_eta(int njet, RVec<int> jet_isgood, RVec<float> jet_eta){
  if (njet<2) return -999;
  bool sublead = false;
  for (unsigned iPart = 0; iPart<jet_eta.size(); iPart++) {
    if (jet_isgood.at(iPart) == 0) continue;
    if (sublead == true) return jet_eta.at(iPart);
    sublead = true;
  }
  return -999;
}
float get_j2_phi(int njet, RVec<int> jet_isgood, RVec<float> jet_phi){
  if (njet<2) return -999;
  bool sublead = false;
  for (unsigned iPart = 0; iPart<jet_phi.size(); iPart++) {
    if (jet_isgood.at(iPart) == 0) continue;
    if (sublead == true) return jet_phi.at(iPart);
    sublead = true;
  }
  return -999;
}
float get_j2_m(int njet, RVec<int> jet_isgood, RVec<float> jet_m){
  if (njet<2) return -999;
  bool sublead = false;
  for (unsigned iPart = 0; iPart<jet_m.size(); iPart++) {
    if (jet_isgood.at(iPart) == 0) continue;
    if (sublead == true) return jet_m.at(iPart);
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

float get_deltaR_photon_j1(RVec<float>jet_isgood,RVec<float>jet_pt,RVec<float>jet_eta,RVec<float>jet_phi,RVec<float>photon_eta,RVec<float>photon_phi){
  for (unsigned iPart = 0; iPart<jet_pt.size(); iPart++) {
    if (jet_isgood.at(iPart)) {
      return deltaR(photon_eta.at(0), photon_phi.at(0), jet_eta.at(iPart), jet_phi.at(iPart));
    }
  }
  return -999;
}
float get_deltaR_photon_j2(RVec<float>jet_isgood,RVec<float>jet_pt,RVec<float>jet_eta,RVec<float>jet_phi,RVec<float>photon_eta,RVec<float>photon_phi){
  bool sublead = false;
  for (unsigned iPart = 0; iPart<jet_pt.size(); iPart++) {
    if (jet_isgood.at(iPart)) {
      if (sublead == false) sublead = true;
      else return deltaR(photon_eta.at(0), photon_phi.at(0), jet_eta.at(iPart), jet_phi.at(iPart));
    }
  }
  return -999;
}

float get_min_deta_llphoton_j(RVec<float>jet_isgood,RVec<float>jet_eta,RVec<float>llphoton_eta){
  bool sublead = false;
  double lead_eta = 0;
  double sublead_eta = -999;
  for (unsigned iPart = 0; iPart<jet_eta.size(); iPart++) {
    if (jet_isgood.at(iPart)) {
      if (sublead == false) {
        sublead = true;
        lead_eta = jet_eta.at(iPart);
      } else {
        sublead_eta = jet_eta.at(iPart);
        break;
      }
    }
  }
  if (sublead_eta > -999) {
    double deta_lead = fabs(llphoton_eta.at(0) - lead_eta);
    double deta_sublead = fabs(llphoton_eta.at(0) - sublead_eta);
    return min(deta_lead, deta_sublead);
  } else return -999;
}
float get_min_dphi_llphoton_j(RVec<float>jet_isgood,RVec<float>jet_phi,RVec<float>llphoton_phi){
  bool sublead = false;
  double lead_phi = 0;
  double sublead_phi = -999;
  for (unsigned iPart = 0; iPart<jet_phi.size(); iPart++) {
    if (jet_isgood.at(iPart)) {
      if (sublead == false) {
        sublead = true;
        lead_phi = jet_phi.at(iPart);
      } else {
        sublead_phi = jet_phi.at(iPart);
        break;
      }
    }
  }
  if (sublead_phi > -999) {
    double dphi_lead = fabs(TVector2::Phi_mpi_pi(llphoton_phi.at(0) - lead_phi));
    double dphi_sublead = fabs(TVector2::Phi_mpi_pi(llphoton_phi.at(0) - sublead_phi));
    return min(dphi_lead, dphi_sublead);
  } else return -999;
}

float get_zeppenfeld_system(RVec<float>jet_eta,RVec<float>jet_isgood,RVec<float>llphoton_eta){
  bool sublead = false;
  double lead_eta = 0;
  double sublead_eta = -999;
  for (unsigned iPart = 0; iPart<jet_eta.size(); iPart++) {
    if (jet_isgood.at(iPart)) {
      if (sublead == false) {
        sublead = true;
        lead_eta = jet_eta.at(iPart);
      } else {
        sublead_eta = jet_eta.at(iPart);
        break;
      }
    }
  }
  if (sublead_eta > -999) {
    double denom = fabs(lead_eta - sublead_eta);
    //if (denom<0.0001) denom = 0.0001;
    double zeppenfeld = (llphoton_eta.at(0) - (lead_eta + sublead_eta)/2)/denom;
    if (zeppenfeld>30) zeppenfeld = 30;
    else if (zeppenfeld<-30) zeppenfeld = -30;
    return zeppenfeld;
  } else return -999;
}
float get_zeppenfeld_pt_system(RVec<float>jet_pt,RVec<float>jet_isgood,RVec<float>llphoton_pt){
  bool sublead = false;
  double lead_pt = 0;
  double sublead_pt = -999;
  for (unsigned iPart = 0; iPart<jet_pt.size(); iPart++) {
    if (jet_isgood.at(iPart)) {
      if (sublead == false) {
        sublead = true;
        lead_pt = jet_pt.at(iPart);
      } else {
        sublead_pt = jet_pt.at(iPart);
        break;
      }
    }
  }
  if (sublead_pt > -999) {
    double denom = fabs(lead_pt - sublead_pt);
    //if (denom<0.0001) denom = 0.0001;
    double pt_balance = (llphoton_pt.at(0) - (lead_pt + sublead_pt)/2)/denom;
    if (pt_balance>30) pt_balance = 30;
    else if (pt_balance<-30) pt_balance = -30;
    return pt_balance;
  } else return -999;
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

float get_dijet(int njet, float dijet_var){
  if (njet<=1) return -999;
  return dijet_var;
}

float get_monojet(int njet, float monojet_var){
  if (njet==0) return -999;
  return monojet_var;
}

""")

if __name__=='__main__':
  #ROOT.EnableImplicitMT() # Turn off to keep ntuple event sequence
  start_time = time.time()
 
  years = ["2016APV","2016","2017","2018"]
  #years = ["2016"]

  #make n-tuples
  defines = [
       # Object variables
       ('y_pt','photon_pt[0]'),     
       ('y_eta','photon_eta[0]'),
       ('y_phi','photon_phi[0]'),     
       ('y_mva','photon_idmva[0]'),
       ('y_id80','photon_id80[0]'),
       ('y_res','photon_energyErr[0]/photon_pt[0]/cosh(photon_eta[0])'),

       ('l1_pt','get_l1_pt(el_pt,mu_pt,ll_lepid,ll_i1,ll_i2)'),
       ('l1_eta','get_l1_eta(el_pt,el_eta,mu_pt,mu_eta,ll_lepid,ll_i1,ll_i2)'),
       ('l1_phi','get_l1_phi(el_pt,el_phi,mu_pt,mu_phi,ll_lepid,ll_i1,ll_i2)'),

       ('l2_pt','get_l2_pt(el_pt,mu_pt,ll_lepid,ll_i1,ll_i2)'),
       ('l2_eta','get_l2_eta(el_pt,el_eta,mu_pt,mu_eta,ll_lepid,ll_i1,ll_i2)'),
       ('l2_phi','get_l2_phi(el_pt,el_phi,mu_pt,mu_phi,ll_lepid,ll_i1,ll_i2)'),

       ('j1_pt','get_j1_pt(njet,jet_isgood, jet_pt)'),
       ('j1_eta','get_j1_eta(njet,jet_isgood, jet_eta)'),
       ('j1_phi','get_j1_phi(njet,jet_isgood, jet_phi)'),
       ('j1_m','get_j1_m(njet,jet_isgood,jet_m)'),

       ('j2_pt','get_j2_pt(njet,jet_isgood,jet_pt)'),
       ('j2_eta','get_j2_eta(njet,jet_isgood,jet_eta)'),
       ('j2_phi','get_j2_phi(njet,jet_isgood,jet_phi)'),
       ('j2_m','get_j2_m(njet,jet_isgood,jet_m)'),

       # Reconstructed variables
       ('leplep_pt', 'll_pt[0]'),
       ('leplep_eta', 'll_eta[0]'),
       ('leplep_phi', 'll_phi[0]'),
       ('leplep_m', 'll_m[0]'),
       ('leplep_flavor', 'get_flavor(ll_lepid)'),
       ('leplep_charge', 'll_charge[0]'),

       ('lly_pt','llphoton_pt[0]'),
       ('lly_eta','llphoton_eta[0]'),
       ('lly_phi','llphoton_phi[0]'),
       ('lly_m','llphoton_m[0]'),

       # General system varaibles
       ('ht','H_t(photon_pt,photon_eta,photon_phi,el_pt,el_eta,el_phi,mu_pt,mu_eta,mu_phi,ll_lepid,ll_i1,ll_i2,jet_pt,jet_eta,jet_phi,jet_m)'),
       ('st','get_st(photon_pt,photon_sig,el_pt,el_sig,mu_pt,mu_sig,jet_pt,jet_isgood)'),

       # BDT variables
       ('y_ptmass','photon_pt[0]/llphoton_m[0]'),      
       ('yl_drmin','photon_drmin[0]'),
       ('yl_drminmass','photon_drmin[0]/llphoton_m[0]'),
       ('yl_drmax','get_max_dr(photon_eta,photon_phi,el_eta,el_phi,mu_eta,mu_phi,ll_lepid,ll_i1,ll_i2)'),
       ('lly_ptmass','llphoton_pt[0]/llphoton_m[0]'),
       ('lly_ptt', 'get_llg_ptt(photon_pt, photon_eta, photon_phi, llphoton_pt, llphoton_eta, llphoton_phi, ll_pt, ll_eta, ll_phi)'),
       ('lly_ptthig19','llphoton_pTt2[0]'),
       ('cosTheta','llphoton_cosTheta[0]'),
       ('costheta','llphoton_costheta[0]'),
       ('phi','llphoton_psi[0]'),

       # SingleJet BDT variables
       ('yj1_deta', 'get_yj1_deta(photon_eta[0], njet, jet_isgood, jet_eta)'),
       ('llyj_dphi', 'get_llyj_dphi(llphoton_phi[0], njet, jet_isgood, jet_phi)'),
       ('llyj1_ptbal', 'get_llyj1_ptbal(ll_pt[0], ll_eta[0], ll_phi[0], photon_pt[0], photon_eta[0], photon_phi[0], njet, jet_isgood, jet_pt, jet_eta, jet_phi)'),

       # DIJET BDT VARIABLEs
       ('jj_dphi','dijet_dphi'),
       ('jj_deta','dijet_deta'),
       ('jj_m','dijet_m'),
       ('yj1_dr','get_deltaR_photon_j1(jet_isgood,jet_pt,jet_eta,jet_phi,photon_eta,photon_phi)'),
       ('yj2_dr','get_deltaR_photon_j2(jet_isgood,jet_pt,jet_eta,jet_phi,photon_eta,photon_phi)'),
       ('yjj_zep','get_dijet(njet,photon_zeppenfeld[0])'),
       ('yj_drmin','get_dijet(njet,photon_jet_mindr[0])'),
       ('llyjj_dphi','get_dijet(njet,llphoton_dijet_dphi[0])'),
       ('llyjj_ptbal','get_dijet(njet,llphoton_dijet_balance[0])'),
       ('llyjj_zep','get_zeppenfeld_system(jet_eta,jet_isgood,llphoton_eta)'),
       ('llyjj_zeppt','get_zeppenfeld_pt_system(jet_pt,jet_isgood,llphoton_pt)'),
       ('llyj_detamin','get_min_deta_llphoton_j(jet_isgood,jet_eta,llphoton_eta)'),
       ('llyj_dphimin','get_min_dphi_llphoton_j(jet_isgood,jet_phi,llphoton_phi)'),

       # system variables
       ('lly_res', 'sqrt(llphoton_l1_masserr[0]*llphoton_l1_masserr[0]+llphoton_l2_masserr[0]*llphoton_l2_masserr[0]+llphoton_ph_masserr[0]*llphoton_ph_masserr[0])'),

       # event variables
       ('event_number','event'),
       ('trigger','get_trigger(ll_lepid,nel,el_sig,el_pt,HLT_Ele27_WPTight_Gsf,HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ,HLT_Ele32_WPTight_Gsf_L1DoubleEG,HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL,HLT_Ele32_WPTight_Gsf,HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL,nmu,mu_sig,mu_pt,HLT_IsoMu24 || HLT_IsoTkMu24,HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL || HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ || HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL || HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ,HLT_IsoMu27,HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8 || HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8,HLT_IsoMu24,HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8, year)'),
       ('pass_filter', 'pass_goodv&&pass_cschalo_tight&&pass_hbhe&&pass_hbheiso&&pass_ecaldeadcell&&pass_badpfmu&&pass_badpfmudz&&pass_hfnoisyhits&&pass_eebadsc&&pass_badcalib'),

       # Truth variables
       ('tm_jets','truth_matched_bool(mc_pt,mc_eta,mc_phi,mc_mass,mc_id,mc_momidx,jet_pt,jet_eta,jet_phi,jet_m,jet_isgood)'),
       ('tru_leplep_m', 'get_tru_leplep_m(mc_id,mc_status,mc_mom,mc_pt,mc_eta,mc_phi,mc_mass)'),

       # Experimental variables
       ('lly_dphi','llphoton_dphi[0]'),
       ('llyjj_dr','get_Zyjj_dr(ll_pt, ll_eta, ll_phi, ll_m, photon_pt, photon_eta, photon_phi, dijet_pt, dijet_eta, dijet_phi, dijet_m)'),
       ('jj_dr','dr_jj(mc_pt,mc_eta,mc_phi,mc_mass,mc_id,mc_momidx,jet_pt,jet_eta,jet_phi,jet_m,jet_isgood)'),
       ('l1j_drmin','drmin_l1j(ll_lepid,el_eta,el_phi,mu_eta,mu_phi,jet_eta,jet_phi)'),
       ('l2j_drmin','drmin_l2j(ll_lepid,el_eta,el_phi,mu_eta,mu_phi,jet_eta,jet_phi)'),
       ('yj_drmax','drmax_yj(mc_pt,mc_eta,mc_phi,mc_mass,mc_id,mc_momidx,jet_pt,jet_eta,jet_phi,jet_m,jet_isgood,photon_eta,photon_phi)'),
       ]
 
  # Select only the needed branches to reduce space
  branches = ['y_pt', 'y_eta', 'y_phi', 'y_mva', 'y_id80', 'y_res']
  branches.extend(['l1_pt', 'l1_eta', 'l1_phi', 'l2_pt', 'l2_eta', 'l2_phi'])
  branches.extend(['j1_pt', 'j1_eta', 'j1_phi', 'j1_m', 'j2_pt', 'j2_eta', 'j2_phi', 'j2_m'])
  branches.extend(['leplep_pt', 'leplep_eta', 'leplep_phi', 'leplep_m', 'leplep_flavor', 'leplep_charge'])
  branches.extend(['lly_pt', 'lly_eta', 'lly_phi', 'lly_m'])
  branches.extend(['ht', 'st'])
  branches.extend(['y_ptmass', 'yl_drmin', 'yl_drminmass', 'yl_drmax'])
  branches.extend(['lly_ptmass', 'lly_ptt', 'lly_ptthig19', 'cosTheta', 'costheta', 'phi'])
  branches.extend(['yj1_deta', 'llyj_dphi', 'llyj1_ptbal'])
  branches.extend(['jj_dphi', 'jj_deta', 'jj_m', 'yj1_dr', 'yj2_dr', 'yjj_zep', 'yj_drmin', 'llyjj_dphi'])
  branches.extend(['llyjj_ptbal', 'llyjj_zep', 'llyjj_zeppt', 'llyj_detamin', 'llyj_dphimin'])
  branches.extend(['lly_res', 'event_number', 'trigger', 'pass_filter', 'tm_jets', 'tru_leplep_m'])
  branches.extend(['lly_dphi', 'llyjj_dr', 'jj_dr', 'l1j_drmin', 'l2j_drmin', 'yj_drmax'])
  branches.extend(['njet', 'nlep','nbdfm', 'nllphoton', 'met', 'npv'])
  branches.extend(['use_event', 'year', 'luminosity', 'w_lumiXyear', 'weightXyear', 'type'])

  # Make n-tuples
  cuts = ['trigger', 'pass_filter', 'nllphoton>=1', 'use_event']

  names = 'ggfvbf_ntuples'
  base_dir  = '/net/cms11/cms11r0/pico/NanoAODv9/htozgamma_kingscanyon_v1/'
  pico_type = '/mc/merged_zgmc_llg/'
  sig_samples = ['*GluGluHToZG_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8*.root','*VBFHToZG_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8*.root']
  bkg_samples = ['*ZGToLLG_01J_5f_lowMLL_lowGPt_TuneCP5_13TeV-amcatnloFXFX-pythia8*.root','*DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8*.root','*ZGamma2JToGamma2L2J_EWK_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8*.root']
  print([base_dir + year + pico_type + sig for sig in sig_samples for year in years])
  print([base_dir + year + pico_type  + bkg for bkg in bkg_samples for year in years])

  # Make signal ntuples
  for sig in sig_samples:
    dataset_name=sig.replace('*','').replace('.root','')
    write_ntuples([base_dir + year + pico_type + sig for year in years],  cuts,  
             (names + '_' + dataset_name + '_' + ''.join(years) + '.root'),  defines,  'tree',  branches)

  # Make bkg ntuples               
  for bkg in bkg_samples:
    dataset_name=bkg.replace('*','').replace('.root','')
    write_ntuples([base_dir + year + pico_type  + bkg for year in years],  cuts,
             (names + '_' + dataset_name + '_' + ''.join(years) + '.root'),  defines,  'tree',  branches)

  elapsed_time = time.time() - start_time
  print(f'Elapsed time: {elapsed_time}')
