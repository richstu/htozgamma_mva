# HToZGamma MVA ReadMe

# Environment requirements
- Above ROOT 6.26
- uproot
- numpy

## (Optional) Install python environment for gpu training (For cms37 UCSB server)
source set_env.sh.gpu
python3 -m venv py-env
source py-env/bin/activate
pip3 install --upgrade pip
pip3 install torch torchvision torchaudio
pip3 install uproot numpy scikit-learn matplotlib
pip3 install jupyterlab xgboost slugify tensorboard shap

## Setup python environment on UCSB servers
- For cms11
```source set_env.sh```
- For cms37 (GPU training)
```source set_env.sh.gpu
source py-env/bin/activate```

# Setup folders
1. Run ```folder_setup.sh``` to make empty folders

# Producing ntuples:
1. Produce ntuples from picos. 
  - ggf_ntuple_producer.py
  - vbf_ntuple_producer.py
2. Make training, validation, and testing ntuples. Apply cut for training.
  - ggf_mva_ntuple_producer.py
  - vbf_mva_ntuple_producer.py

# Procedure for making BDTs:
1. Run the BDT producer
   - ggf_train_tmva_hig19014.py
   - vbf_train_tmva_hig19014.py

# Procedure for measuring performance of MVAs
TODO
1. Measure significance
2. Measure AUC
3. Measure correlation with lly_m
4. Measure overtraining
