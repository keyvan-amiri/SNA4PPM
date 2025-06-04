## LS-ICE
We used the LS-ICE [repository](https://github.com/bjornragu/LS-ICE) to extract inter-case features. Following the original paper, we then used the CRTP [repository](https://github.com/bjornragu/CRTP-LSTM) to train data-aware LSTM models that incorporate both intra- and inter-case features. To integrate the original implementation into our evaluation pipeline, we made several modifications—all of which are included in this repository. 

## PGTNet
We followed the instructions from the PGTNet [repository](https://github.com/keyvan-amiri/PGTNet). However, since we adapted the baseline for additional event logs not covered in the original study, we included all necessary scripts and configuration files in this repository to enable full replication of our experiments.
