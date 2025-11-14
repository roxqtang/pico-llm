\# Part D: Normalization Comparison



This folder contains the code for Part D of the project.  

The goal is to compare three normalization methods:



\- RMSNorm (custom implementation)

\- LayerNorm (PyTorch)

\- NoNorm (no normalization)



A small 12-layer MLP is trained on randomly generated dummy data.  

The loss curves are used to observe the stability of each method.



\## Files



\- `partD\_rmsnorm\_experiment.py` — main script for running the experiment  

\- `partD\_rmsnorm\_vs\_layernorm\_loss.png` — loss curve result  



\## How to run



```bash

python partD\_rmsnorm\_experiment.py



