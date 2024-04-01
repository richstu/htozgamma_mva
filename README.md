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

# Producing ntuples:
1. Produce ntuples from picos. 
  - `ggf_ntuple_producer.py`
  - `vbf_ntuple_producer.py`
2. Make training, validation, and testing ntuples. Apply cut for training. Add additional variables.
  - `ggf_mva_ntuple_producer.py`
  - `vbf_mva_ntuple_producer.py`
3. Make training, validation, and testing ntuples with loose photon ID.
  - `ggf_mva_ntuple_producer.py -b hig19014`

# Procedure for training BDTs:
Run the BDT trainer
 - `ggf_train_tmva_standard.py`
 - `ggf_train_xgboost_standard.py`
 - `vbf_train_tmva_standard.py`

# Procedure for measuring performance of MVAs
1. Edit the `ggf_measure_performance.py` to include the trained MVA
2. Run the `ggf_measure_performance.py` script
