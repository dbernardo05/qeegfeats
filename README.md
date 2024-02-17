# qeegfeats

Pipeline to generate QEEG features for various EEG classification/prediction/forecasting tasks.

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
