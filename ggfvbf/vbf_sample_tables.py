#!/usr/bin/env python3
import ROOT

# Args
#   samples[sample] = [filename, [tree name], sample cut]
#   cut_string
#   weight_name
#   obs_hist_parameters = [observable name, nbins, min, max]
# Returns
#   sample_yields[sample][tree name] = [cut&no_weight,cut&weight]
def get_sample_yields(samples, cut_string, weight_name, obs_hist_parameters):
  sample_yields = {}
  for sample_name in samples:
    sample_filename = samples[sample_name][0]
    sample_cut = samples[sample_name][2]
    for tree_name in samples[sample_name][1]:
      hist_name = f'hist_{obs_hist_parameters[0]}'
      hist = ROOT.TH1D(hist_name, hist_name, obs_hist_parameters[1], obs_hist_parameters[2], obs_hist_parameters[3])
      chain = ROOT.TChain(tree_name)
      chain.Add(sample_filename)
      chain.Draw(f'{obs_hist_parameters[0]}>>{hist_name}', f'({cut_string}&&{sample_cut})*{weight_name}', 'goff')
      if sample_name not in sample_yields: sample_yields[sample_name] = {}
      if tree_name not in sample_yields[sample_name]: sample_yields[sample_name][tree_name] = []
      sample_yields[sample_name][tree_name] = [hist.GetEntries(), hist.GetSum()]
  return sample_yields

# rows = [[value, value, ...], next row]
# row_align = 'cccc'
def make_tex_table(rows, row_align, output_filename):
  tex_string = r'''\documentclass[10pt,oneside]{report}
\usepackage{graphicx,xspace,amssymb,amsmath,colordvi,colortbl,verbatim,multicol}
\usepackage{multirow, rotating}

\usepackage[active,tightpage]{preview}

\usepackage{siunitx}
\sisetup{round-mode = figures, round-precision=2}

\renewcommand{\arraystretch}{1.1}

\begin{document}
\begin{preview}
'''
  tex_string += f'\\begin{{tabular}}{{ {row_align} }}\n'
  for row in rows:
    for item in row:
      tex_string += f'{item} &'
    tex_string = tex_string[:-1]+'\\\\\n'
  tex_string += r'\end{tabular}'
  tex_string += r'''\end{preview}
\end{document}
'''
  with open(output_filename,'w') as output_file:
    output_file.write(tex_string)
  print(f'Wrote to {output_filename}')

def get_cut_yields(samples, tree_name, weight_name, baseline_cut, feature_cut_dict): 
  feature_yields = {}
  hist_name = f'hist_{obs_hist_parameters[0]}'
  hist = ROOT.TH1D(hist_name, hist_name, obs_hist_parameters[1], obs_hist_parameters[2], obs_hist_parameters[3])
  for sample_name in samples:
    sample_filename = samples[sample_name][0]
    chain = ROOT.TChain(tree_name)
    chain.Add(sample_filename)
    for feature_cut_name in feature_cut_dict:
      feature_cut = feature_cut_dict[feature_cut_name]
      chain.Draw(f'{obs_hist_parameters[0]}>>{hist_name}', f'({baseline_cut}&&{feature_cut})*{weight_name}', 'goff')
      if sample_name not in feature_yields: feature_yields[sample_name] = {}
      if feature_cut_name not in feature_yields[sample_name]: feature_yields[sample_name][feature_cut_name] = []
      feature_yields[sample_name][feature_cut_name] = [hist.GetEntries(), hist.GetSum()]
  return feature_yields

if __name__ == '__main__':

  # obs_hist_parameters = [observable name, nbins, min, max]
  obs_hist_parameters = ['lly_m', 160, 100, 180]

  # inclusive_samples[sample] = [filename, [tree name], sample_cut]
  inclusive_samples = {'SMZG': ['ntuples/vbf_ntuples_ZGToLLG_01J_5f_lowMLL_lowGPt_TuneCP5_13TeV-amcatnloFXFX-pythia8_2016APV201620172018.root', ['tree'], '1'],
             'DYJets': ['ntuples/vbf_ntuples_DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_2016APV201620172018.root', ['tree'], '1'],
             'ZG(VBS)': ['ntuples/vbf_ntuples_ZGamma2JToGamma2L2J_EWK_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8_2016APV201620172018.root', ['tree'], '1'],
             'GGF': ['ntuples/vbf_ntuples_GluGluHToZG_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8_2016APV201620172018.root', ['tree'], '1'],
             'VBF': ['ntuples/vbf_ntuples_VBFHToZG_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8_2016APV201620172018.root', ['tree'], '1']}
  #inclusive_samples = {'SMZG': ['ntuples/ggf_ntuples_ZGToLLG_01J_5f_lowMLL_lowGPt_TuneCP5_13TeV-amcatnloFXFX-pythia8_2016APV201620172018.root', ['tree']]}
  # Get yields for inclusive sample table
  baseline_cut = '(trigger>0) & (pass_filter>0) & (use_event>0) & (y_id80>0) & (y_pt/lly_m>15./110) & (leplep_m>80) & (leplep_m<100) & (lly_m+leplep_m>185) & (lly_m>100) & (lly_m<180) & (njet>=2) & (nlep==2) & (nbdfm==0)'
  #baseline_cut = '(trigger>0) & (pass_filter>0) & (use_event>0) & (y_id80>0) & (y_pt/lly_m>15./110) & (leplep_m>80) & (leplep_m<100) & (lly_m+leplep_m>185) & (lly_m>100) & (lly_m<180)'
  baseline_weight = 'w_lumiXyear'
  #baseline_weight = 'weightXyear'

  ## Make cutflow table
  #cutflow_dict = {'overlap removal, trigger, filter, $N_{\ell\ell\gamma}>=1$': '(trigger>0) && (pass_filter>0) && (use_event>0) && (y_id80>0)',
  #                r'$80<m_{\ell\ell}<100$':'(leplep_m>80) && (leplep_m<100)',
  #                r'$p_{T}^{\gamma}>15/110$': 'y_pt/lly_m>15./110',
  #                r'$m_{\ell\ell}+m_{\ell\ell\gamma}<185$': 'lly_m+leplep_m>185',
  #                r'$100<m_{\ell\ell\gamma}<180$': '(lly_m>100) && (lly_m<180)',
  #                r'$N_{\ell}=2$': 'nlep==2',
  #                r'$N_{\text{jet}}>=2$':'njet>=2',
  #                r'$N_{\text{bjet}}^{M}=0$':'nbdfm==0'
  #                }
  #cutflow_cut_dict = {}
  #for icut, cut_name in enumerate(cutflow_dict):
  #  if icut == 0: cut = cutflow_dict[cut_name]
  #  else: cut = f'({cut})&&({cutflow_dict[cut_name]})'
  #  cutflow_cut_dict[cut_name] = cut
  ## cutflow_yields[sample][feature] = [no_weight, weight]
  #cutflow_yields = get_cut_yields(inclusive_samples, 'tree', baseline_weight, '1', cutflow_cut_dict)
  ## inclusive_sample_yields[sample][tree name] = [baseline&no_weight, baseline&weight]
  #inclusive_sample_yields = get_sample_yields(inclusive_samples, baseline_cut, baseline_weight, obs_hist_parameters)
  ## Make tex table using yields
  #cutflow_table = []
  #cutflow_table.append(['Variable','SMZG', 'DYJets', 'ZG(VBS)', 'GGF', 'VBF'])
  #for cut_name in cutflow_cut_dict:
  #  cutflow_table.append([cut_name, cutflow_yields['SMZG'][cut_name][0], cutflow_yields['DYJets'][cut_name][0], cutflow_yields['ZG(VBS)'][cut_name][0], cutflow_yields['GGF'][cut_name][0], cutflow_yields['VBF'][cut_name][0] ])
  #cutflow_table.append(['with weight',f"{inclusive_sample_yields['SMZG']['tree'][1]:.2f}", f"{inclusive_sample_yields['DYJets']['tree'][1]:.2f}", f"{inclusive_sample_yields['ZG(VBS)']['tree'][1]:.2f}", f"{inclusive_sample_yields['GGF']['tree'][1]:.2f}", f"{inclusive_sample_yields['VBF']['tree'][1]:.2f}"])
  #make_tex_table(cutflow_table, 'cccccc', 'tables/cutflow_table.tex')

  ## Get yields for input feature table
  #feature_cut_dict = {r'Baseline': '1',
  #                    r'Baseline \& $|\text{cos}\Theta|<0.8$': 'fabs(cosTheta)<0.8',
  #                    r'Baseline \& $|\text{cos}\theta|>0.5$': 'fabs(costheta)>0.5',
  #                    r'Baseline \& $|\phi|<2$': 'fabs(phi)<2',
  #                    r'Baseline \& $p_{Tt}^{\ell\ell\gamma}>50$': 'lly_ptt>50',
  #                    r'Baseline \& $\Delta R_{\text{min}}(\gamma,\ell)<1.4$': 'yl_drmin<1.4',
  #                    r'Baseline \& $\Delta R_{\text{max}}(\gamma,\ell)<2$': 'yl_drmax<2',
  #                    r'Baseline \& $|\eta_{\ell_{1}}|<1.5$': 'fabs(l1_eta)<1.5',
  #                    r'Baseline \& $|\eta_{\ell_{2}}|<1.5$': 'fabs(l2_eta)<1.5',
  #                    r'Baseline \& $|\eta_{\gamma}|<2$': 'fabs(y_eta)<2',
  #                    r'Baseline \& $\text{MVAID}_{\gamma}>0.4$': 'y_mva>0.5',
  #                    r'Baseline \& $\text{Resolution}_{\gamma} / p_{T}^{\gamma}<0.1$': 'y_res<0.1',
  #                    r'Baseline \& $|\eta_{j1}-\eta_{j2}|>3$': 'jj_deta>3',
  #                    r'Baseline \& $|\phi_{j1}-\phi_{j2}|<2.5$': 'abs(jj_dphi)<2.5',
  #                    r'Baseline \& $p_{T}^{j1}>50$': 'j1_pt>50',
  #                    r'Baseline \& $p_{T}^{j2}>40$': 'j2_pt>40',
  #                    r'Baseline \& $\Delta \phi (ll\gamma, jj)>2$': 'abs(llyjj_dphi)>2',
  #                    r'Baseline \& $\Delta R(\gamma,j1)>1$': 'yj1_dr>1',
  #                    r'Baseline \& $\Delta R(\gamma,j2)>1.3$': 'yj2_dr>1.3',
  #                    r'Baseline \& $|\eta_{\gamma}-(\eta_{j1}+\eta_{j2})/2|<1.4$': 'yjj_zep<1.4',
  #                    r'Baseline \& $\sum_{\ell\ell,j1,j2} p_{T}/Mag. (\sum_{\ell\ell,j1,j2} p_{T})<0.2$': 'llyjj_ptbal<0.2',
  #                    }
  ## feature_yields[sample][feature] = [no_weight, weight]
  #feature_yields = get_cut_yields(inclusive_samples, 'tree', baseline_weight, baseline_cut, feature_cut_dict)
  ## Make tex table using yields
  #feature_table = []
  #feature_table.append(['Variable','SMZG', 'DYJets', 'ZG(VBS)', 'GGF', 'VBF'])
  #for feature_name in feature_cut_dict:
  #  feature_table.append([feature_name, feature_yields['SMZG'][feature_name][0], feature_yields['DYJets'][feature_name][0], feature_yields['ZG(VBS)'][feature_name][0],feature_yields['GGF'][feature_name][0], feature_yields['VBF'][feature_name][0]])
  #make_tex_table(feature_table, 'cccccc', 'tables/feature_table.tex')

  # Make 3fold sample table
  # nfold_samples[sample] = [filename, [train tree name, validation tree name, test tree name]]
  nfold_samples = {'SMZG': ['ntuples/ggfvbf_tight_2j_shape_decorr_weight_ntuples.root', ['train_tree', 'eval_tree', 'test_tree'], 'sampleID==1'],
                  'DYJets': ['ntuples/ggfvbf_tight_2j_shape_decorr_weight_ntuples.root', ['train_tree', 'eval_tree', 'test_tree'], 'sampleID==2'],
                  'VBSZG': ['ntuples/ggfvbf_tight_2j_shape_decorr_weight_ntuples.root', ['train_tree', 'eval_tree', 'test_tree'], 'sampleID==3'],
                  'GGF': ['ntuples/ggfvbf_tight_2j_shape_decorr_weight_ntuples.root', ['train_tree', 'eval_tree', 'test_tree'], 'sampleID==4'],
                  'VBF': ['ntuples/ggfvbf_tight_2j_shape_decorr_weight_ntuples.root', ['train_tree', 'eval_tree', 'test_tree'], 'sampleID==5'],}
  # nfold_sample_yields[sample][tree name] = [baseline&no_weight, baseline&weight]
  nfold_sample_yields = get_sample_yields(nfold_samples, cut_string='1', weight_name='w_lumiXyear', obs_hist_parameters=obs_hist_parameters)
  # Make tex table using yields
  nfold_sample_table = []
  samples = ('SMZG', 'DYJets', 'VBSZG', 'GGF', 'VBF')
  nfold_sample_table.append(['sample', 'cut$\\times$weight'] + [sample for sample in samples])
  nfold_sample_table.append(['train', '(baseline\&$120<m_{\ell\ell\gamma}<130$)$\\times$weight'] + [f"{nfold_sample_yields[sample]['train_tree'][1]:.2f}" for sample in samples]),
  nfold_sample_table.append(['train', '(baseline\&$120<m_{\ell\ell\gamma}<130$)$\\times$1'] + [nfold_sample_yields[sample]['train_tree'][0] for sample in samples])
  nfold_sample_table.append(['eval',  '(baseline\&$120<m_{\ell\ell\gamma}<130$)$\\times$weight'] + [f"{nfold_sample_yields[sample]['eval_tree'][1]:.2f}" for sample in samples]),
  nfold_sample_table.append(['eval',  '(baseline\&$120<m_{\ell\ell\gamma}<130$)$\\times$1'] + [nfold_sample_yields[sample]['eval_tree'][0] for sample in samples])
  nfold_sample_table.append(['test',  '(baseline\&$120<m_{\ell\ell\gamma}<130$)$\\times$weight'] + [f"{nfold_sample_yields[sample]['test_tree'][1]:.2f}" for sample in samples]),
  nfold_sample_table.append(['test',  '(baseline\&$120<m_{\ell\ell\gamma}<130$)$\\times$1'] + [nfold_sample_yields[sample]['test_tree'][0] for sample in samples])
  make_tex_table(nfold_sample_table, 'ccccccc', 'tables/nfold_vbf_sample_table.tex')

