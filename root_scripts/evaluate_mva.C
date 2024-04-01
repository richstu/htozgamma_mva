// Referenced from: https://github.com/mcoshiro/small_phys_utils/blob/main/src/distribution_analyzer.cpp
#include <iostream>
#include "TChain.h"
#include "TH1D.h"

/**
 * @brief Returns area under ROC curve for given distributions
 *
 * @param signal            pointer to ROOT histogram of signal
 * @param background        pointer to ROOT histogram of background
 */
 tuple<float, float, vector<float>, vector<float>, vector<float>> 
 get_roc_auc(TH1D* signal, TH1D* background) {
    // results = (auc, csi, bins, areas, signis)
    tuple<float, float, vector<float>, vector<float>, vector<float>> results;
    //integrate where measure is %bkg accepted
    int hist_nbins = (signal->GetNbinsX())+2;
    float total_bkg = background->Integral();
    float total_sig = signal->Integral();
    vector<float> areas(hist_nbins-1);
    vector<float> signis(hist_nbins-1);
    vector<float> bins(hist_nbins-1);
    float integrated_area = 0;
    float casi = 0;
    float prev_bkg = 1.0;
    float prev_sig = 1.0;
    float bkg_accepted = 0.0;
    float sig_accepted = 0.0;
    for (int ibin = 1; ibin < hist_nbins; ibin++) {
      bkg_accepted = (background->Integral(ibin,hist_nbins))/total_bkg;
      sig_accepted = (signal->Integral(ibin,hist_nbins))/total_sig;
      bins[ibin-1] = signal->GetBinCenter(ibin);
      integrated_area += (sig_accepted+prev_sig)/2.0*(prev_bkg-bkg_accepted);
      //cout<<ibin<<" "<<bins[ibin-1]<<" sig:"<<(sig_accepted+prev_sig)/2.0<<" bkg:"<<prev_bkg-bkg_accepted<<" s/b:"<<(sig_accepted+prev_sig)/2.0*(prev_bkg-bkg_accepted)<<endl;
      areas[ibin-1] = (sig_accepted+prev_sig)/2.0*(prev_bkg-bkg_accepted);
      if (prev_bkg>bkg_accepted) {
        casi += (sig_accepted-prev_sig)*(sig_accepted-prev_sig)/(prev_bkg-bkg_accepted);
        signis[ibin-1] = sqrt((sig_accepted-prev_sig)*(sig_accepted-prev_sig)/(prev_bkg-bkg_accepted));
      }
      prev_bkg = bkg_accepted;
      prev_sig = sig_accepted;
    }
    //std::cout << "CSI = Continuous significance improvement (approximation to significance improvement)." << std::endl;
    //std::cout << "ROC AUC is: " << integrated_area << std::endl;
    //std::cout << "ROC CSI is: " << sqrt(casi) << std::endl;
    get<0>(results) = integrated_area;
    get<1>(results) = sqrt(casi);
    get<2>(results) = bins;
    get<3>(results) = areas;
    get<4>(results) = signis;
    return results;
}

/**
 * @brief Function to find optimized binning cuts given distribution as histograms
 *
 * @param signal            pointer to ROOT histogram of signal
 * @param background        pointer to ROOT histogram of background
 * @param max_nbins         maximum number of bins to consider (ncuts + 1)
 * @param min_signal_yield  minimum signal yield allowed in a bin
 * @param scale             scale factor for signal and background to change effective lumi
 * @param nbins_throw       number of bins to remove from significance calculation
 */
vector<tuple<int, vector<float>, float, float, vector<float>, vector<float>, vector<float>, vector<float> > >
binning_optimizer(TH1D* signal, TH1D* background, int max_nbins, 
    float min_signal_yield, float scale, unsigned nbins_throw) {

  //bin_results = [(nbins, bin_edges, significance, significance_err, significances, significance_errs, signal yield, background yield)]
  vector<tuple<int, vector<float>, float, float, vector<float>, vector<float>, vector<float>, vector<float> > > bin_results;

  int hist_nbins = signal->GetNbinsX();

  // Find optimal cuts when having nbins
  for (int nbins = 1; nbins <= max_nbins; nbins++) {
    //std::cout << "With " << nbins << " bins: " << std::endl;
    if (nbins < 2) {
      //float bin_s = signal->Integral()*scale;
      //float bin_b = background->Integral()*scale;
      Double_t bin_s_err, bin_b_err;
      float bin_s = signal->IntegralAndError(0,hist_nbins+2, bin_s_err);
      float bin_b = background->IntegralAndError(0,hist_nbins+2, bin_b_err);
      //std::cout << "Optimal cuts(sig,bak): No cut(sig:" << bin_s << ", bkg:" << bin_b << ")" << std::endl;
      //std::cout << "Estimated significance: " << bin_s/sqrt(bin_b) << std::endl;

      // Save results
      tuple<int, vector<float>, float, float, vector<float>, vector<float>, vector<float>, vector<float> > bin_result;
      get<0>(bin_result) = 1;
      get<1>(bin_result) = {};
      float bin_significance_square = bin_s*bin_s/bin_b;
      get<2>(bin_result) = sqrt(bin_significance_square);
      get<3>(bin_result) = sqrt(1/bin_significance_square*pow(bin_s/bin_b,2)*pow(bin_s_err,2) + 1/bin_significance_square/4*pow(bin_s/bin_b,4)*pow(bin_b_err,2));
      get<4>(bin_result) = {get<2>(bin_result)};
      get<5>(bin_result) = {get<3>(bin_result)};
      get<6>(bin_result) = {bin_s};
      get<7>(bin_result) = {bin_b};
      bin_results.push_back(bin_result);
      continue;
    }
    float max_significance = 0;
    float max_significance_err = 0;
    std::vector<int> max_sig_cuts;
    vector<float> max_significances(nbins);
    vector<float> max_significance_errs(nbins);
    vector<float> max_signals(nbins);
    vector<float> max_backgrounds(nbins);
    //initialize vector
    //convention: cuts are specified as lowest histogram bin in upper binning
    std::vector<int> cuts(nbins-1,0);
    for (int i = 0; i<(nbins-1); i++) {
      cuts[i] = i+1;
    }
    //loop over all possible combinations of cuts to find optimum 
    bool finished = false;
    while (!finished) {
      //cout<<"Trial: "<<endl;
      //calculate significance for current cuts
      std::vector<int> extended_cuts;
      extended_cuts.push_back(0);
      extended_cuts.insert(extended_cuts.end(),cuts.begin(),cuts.end());
      extended_cuts.push_back(hist_nbins+2);
      float bin_significance_square = 0;
      float bin_significance_err_square = 0;
      float significance = 0;
      float significance_err = 0;
      vector<float> significances(nbins);
      vector<float> significance_errs(nbins);
      vector<float> signals(nbins);
      vector<float> backgrounds(nbins);
      bool can_throw = false;
      if (extended_cuts.size() > nbins_throw+1) can_throw = true;
      for (unsigned i = 0; i<(extended_cuts.size()-1); i++) {
        //cout<<"  extended_cut["<<i<<"]: "<<extended_cuts[i]<<" extended_cuts[i+1]-1:"<<extended_cuts[i+1]-1;
        //float bin_s = signal->Integral(extended_cuts[i],extended_cuts[i+1]-1)*scale;
        //float bin_b = background->Integral(extended_cuts[i],extended_cuts[i+1]-1)*scale;
        //cout<<" sig: "<<bin_s<<" bkg: "<<bin_b<<endl;
        Double_t bin_s_err, bin_b_err;
        float bin_s = signal->IntegralAndError(extended_cuts[i],extended_cuts[i+1]-1, bin_s_err);
        float bin_b = background->IntegralAndError(extended_cuts[i],extended_cuts[i+1]-1, bin_b_err);
        //cout<<" sig: "<<t_bin_s*scale<<" +- "<<bin_s_err*scale<<" bkg: "<<t_bin_b*scale<<" +- "<<bin_b_err*scale<<endl;
        if (bin_s <= min_signal_yield) bin_s = 0;
        if (bin_b <= 0) bin_b = 0.01;
        if (!(can_throw && i<nbins_throw))
          bin_significance_square = bin_s*bin_s/bin_b;
          significance += bin_significance_square;
          significances[i] = sqrt(bin_significance_square);
          bin_significance_err_square = 1/bin_significance_square*pow(bin_s/bin_b,2)*pow(bin_s_err,2) + 1/bin_significance_square/4*pow(bin_s/bin_b,4)*pow(bin_b_err,2);
          significance_err += bin_significance_err_square;
          significance_errs[i] = sqrt(bin_significance_err_square);
          signals[i] = bin_s;
          backgrounds[i] = bin_b;
      }
      significance = sqrt(significance);
      significance_err = sqrt(significance_err);
      if (significance > max_significance) {
        max_significance = significance;
        max_significance_err = significance_err;
        max_sig_cuts = cuts;
        max_significances = significances;
        max_significance_errs = significance_errs;
        max_signals = signals;
        max_backgrounds = backgrounds;
      }
      //debug print cuts
      //std::cout << "Cuts: "; for (int cut : cuts) std::cout << ", " << cut; std::cout << std::endl;
      //iterate to next set of cuts
      int place = 0;
      bool iterated = false;
      while (!iterated) {
        //cout<<"  cuts: ";for (int cut:cuts) cout<<cut<<" "; cout<<endl;
        if (cuts[nbins-2-place] != hist_nbins+1-place) {
          cuts[nbins-2-place]++;
          int place_offset = 1;
          for (int lower_place = place-1; lower_place >= 0; lower_place--) {
            cuts[nbins-2-lower_place] = cuts[nbins-2-place]+place_offset;
            place_offset++;
          }
          iterated = true;
        }
        else {
          if (place == nbins-2) {
            iterated = true;
            finished = true;
          }
          else {
            place += 1;
          }
        }
        //cout<<"  new cuts: ";for (int cut:cuts) cout<<cut<<" "; cout<<endl;
      }//end loop used to increment cuts
    }//end loop over all possible cuts

    ////print optimal cuts
    //cout<<"  max cuts: ";for (int cut:max_sig_cuts) cout<<cut<<" "; cout<<endl;
    //std::cout << "Optimal low_cut~high_cut(signal,bak,signi): ";
    //std::cout << signal->GetBinLowEdge(1) << "~" <<signal->GetBinLowEdge(max_sig_cuts[0]);
    //std::cout << "(" << signal->Integral(1,max_sig_cuts[0]-1)*scale;
    //std::cout << "," << background->Integral(1,max_sig_cuts[0]-1)*scale << ")";
    //bool first = false;
    //for (int i = 0; i<(nbins-1); i++) {
    //  if (!first) std::cout << ", ";
    //  if (i!=nbins-2) std::cout << signal->GetBinLowEdge(max_sig_cuts[i]) << "~" << signal->GetBinLowEdge(max_sig_cuts[i+1]) ;
    //  else std::cout << signal->GetBinLowEdge(max_sig_cuts[i]) << "~" << signal->GetBinLowEdge(hist_nbins)+signal->GetBinWidth(1) ;
    //  int upper_lim = hist_nbins;
    //  if (i != nbins-2) {
    //    upper_lim = max_sig_cuts[i+1]-1;
    //  }
    //  std::cout << "(" << signal->Integral(max_sig_cuts[i],upper_lim)*scale;
    //  std::cout << "," << background->Integral(max_sig_cuts[i],upper_lim)*scale;
    //}
    //std::cout << std::endl;
    //std::cout << "Estimated significance: " << max_significance << "+-"<< max_significance_err << std::endl;
    
    // Save results
    //bin_result = (nbins, bin_edges, significance, significances, signal yield, background yield)

    tuple<int, vector<float>, float, float, vector<float>, vector<float>, vector<float>, vector<float> > bin_result;
    get<0>(bin_result) = nbins;
    vector<float> bin_edges;
    for (int binIndex : max_sig_cuts) bin_edges.push_back(signal->GetBinLowEdge(binIndex));
    get<1>(bin_result) = bin_edges;
    get<2>(bin_result) = max_significance;
    get<3>(bin_result) = max_significance_err;
    get<4>(bin_result) = max_significances;
    get<5>(bin_result) = max_significance_errs;
    get<6>(bin_result) = max_signals;
    get<7>(bin_result) = max_backgrounds;
    bin_results.push_back(bin_result);
  }
  return bin_results;
}

void evaluate_mva() {   
  TChain * chain = new TChain("eval_tree_nocut");
  chain->Add("mva_output/tmva_bdt_hig19014_results.root");
  float mva_min  = chain->GetMinimum("BDT");
  float mva_max  = chain->GetMaximum("BDT");
  TH1D * bkg_hist = new TH1D("bkg_hist","bkg_hist",100, mva_min, mva_max);
  TH1D * sig_hist = new TH1D("sig_hist","sig_hist",100, mva_min, mva_max);
  chain->Draw("BDT>>bkg_hist","(lly_m>122&&lly_m<128&&classID==0)*weightXyear","goff");
  chain->Draw("BDT>>sig_hist","(lly_m>122&&lly_m<128&&classID==1)*weightXyear","goff");
  get_roc_auc(sig_hist, bkg_hist);
  binning_optimizer(sig_hist, bkg_hist, /*max_bins*/5, /*min signal yield*/0.5, /*scale*/1.0, /*nbins_remove*/0);
}
