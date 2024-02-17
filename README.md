# qeegfeats

Pipeline to generate QEEG features for various EEG classification/prediction/forecasting tasks.
The algorithms to generate QEEG features are located in scripts/preproc.py

## Requirements

- Python (3.8 or later)
- CUDA 12.1
- CUDNN 8.5
- Multiple Python packages in requirements.txt
- To install these Python packages, run the following command:

```bash
pip install -r requirements.txt
```
## Installation

```bash
git clone https://github.com/dbernardo05/qeegfeats.git
```
## Operation
- edit batch_features.sh to include your EEG file directory and desired output directory
- then run the following command:

```bash
cd qeegfeats
chmod +x batch_features.sh
```

## Acknowledgements
We deeply appreciate the numerous participants of the Kaggle Seizure Prediction Contest (Brinkmann et al.), who have graciously shared their QEEG algorithms. Many of the algorithms in this code were directly forked from various Kagglers cited in Brinkmann et al.

## References
Brinkmann BH, Wagenaar J, Abbot D, Adkins P, Bosshard SC, Chen M, Tieng QM, He J, Muñoz-Almaraz FJ, Botella-Rocamora P, Pardo J. Crowdsourcing reproducible seizure forecasting in human and canine epilepsy. Brain. 2016 Jun 1;139(6):1713-22.
