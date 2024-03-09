# HToZGamma MVA ReadMe

# Setup folders
1. Run ```folder_setup.sh``` to make empty folders

# Producing ntuples:
1. Setup environment: ```source set_env.sh```
2. Produce ntuples from picos
  - ggf_ntuple_producer.py
  - vbf_ntuple_producer.py
3. Make training, validation, and testing ntuples
  - ggf_mva_ntuple_producer.py

# Procedure for making BDTs:
1. Make ntuples using procedure above
TODO 2. run the BDT producer file with root (e.g. root ggF_BDT_producer.C)
