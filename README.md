# HToZGamma MVA ReadMe

# Environment requirements
- ROOT version >= 6.26
- uproot, numpy
- scikit-learn, matplotlib
- xgboost, shap, torch

# Installing python environment on UCSB servers (CentOS7)
```
source set_env.sh
python3 -m venv py-env
source py-env/bin/activate
pip3 install --upgrade pip
pip3 install uproot numpy scikit-learn matplotlib xgboost shap 
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
```

# Setup python environment on UCSB servers
```
source set_env.sh
source py-env/bin/activate
```

# Setup folders
1. Run `folder_setup.sh` to make empty folders

# Producing ntuples for ggf-targeted and vbf-targeted categories:
1. cd to the `ggfvbf` folder
2. Produce ntuples from picos. 
  - `ggfvbf_ntuple_producer.py`
3. Make training, validation, and testing ntuples. Apply cut for training. Add additional variables.
  - `ggfvbf_mva_ntuple_producer.py -b tight_012j`: For ggf-targted category
  - `ggfvbf_mva_ntuple_producer.py -b tight_2j`: For vbf-targeted category

# Procedure of making yield tables for samples
- `ggf_tables.py` and `vbf_tables.py` produces the below tables
  - inclusive sample table
  - input feature table
  - 3fold sample table

# Procedure for training BDTs:
0. cd to the `ggfvbf` folder
1. Run the BDT trainer
 - `ggfvbf_train_tmva_standard.py -i 0`: ggf-targeted category with 11 variables
 - `ggfvbf_train_tmva_standard.py -i 1`: vbf-targeted category with 21 variables
 - `ggfvbf_train_xgboost_standard.py -i 0`: ggf-targeted category with 11 variables
 - `ggfvbf_train_xgboost_standard.py -i 1`: vbf-targeted category with 21 variables

# Procedure for measuring performance of MVAs
0. cd to the `ggfvbf` folder
1. Run the `ggfvbf_measure_performance.py`. Will need to edit file if BDT is not in `ggfvbf_measure_performance.py`. 
  - `ggfvbf_measure_performance.py -m tmva_bdta_ggfvbf_tight_01j_higwindow_var11`

# Procedure of using peking files
1. cd to the `ggfvbf` folder
2. Download the peking files to `peking_ntuples` directory. (Make the directory)
3. Produce ntuples from peking files.
  - `ggfvbf_peking_ntuple_producer.py`
4. Run the BDT trainer
 - `ggfvbf_train_tmva_standard.py -i 2`: ggf-targeted category with 11 variables with peking ntuples
5. Evaluate trained MVA
  - `ggfvbf_measure_performance.py -m tmva_bdta_peking_higwindow_var11`
