# MFINet
## This repo is built for the paper: A Novel Zero-Shot Remote Sensing Scene Classification Network Based on Multimodal Feature Interaction 
### [<a href="https://doi.org/10.1109/JSTARS.2024.3414499">Paper</a>]

# Getting Started
## Installation
### Step 1: Clone the MFINet repository:
To get started, first clone the MFINet repository and navigate to the project directory:
```
git clone https://github.com/XM-Tan/MFINet.git
cd MFINet
```
### Step 2: Environment Setup:
MFINet recommends setting up a conda environment and installing dependencies via pip. 
Use the following commands to set up your environment:

***Creat and activate a new conda environment***

```
conda create -n MFINet python=3.9
conda activate MFINet
```

***Install Dependencies***

```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

*Install Transformer*

```
pip install huggingface_hub
conda install -c huggingface transformers==4.16.2
```

*Others*

```
pip install scikit-learn
pip install timm
pip install h5py
```

# Model Training and Testing

To train and test MFINet for zero-shot classification on RSSDIVCS, use the following commands for different configurations:

```
python ./main.py --dataset RSSDIVCS --batch_size 6 --manualSeed 42 --xlayer_num 1 --seen_unseen_ratio 6010 --random_num 1
```

# Citation
If it is helpful for your work, please cite this paper:
``` 
@ARTICLE{10557622,
  author={Tan, Xiaomeng and Xi, Bobo and Xu, Haitao and Li, Yunsong and Xue, Changbin and Chanussot, Jocelyn},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={MFINet: A Novel Zero-Shot Remote Sensing Scene Classification Network Based on Multimodal Feature Interaction}, 
  year={2024},
  volume={17},
  number={},
  pages={11670-11684},
  keywords={Visualization;Semantics;Feature extraction;Remote sensing;Vectors;Training;Scene classification;Cross-modal feature fusion (CMFF);improved Transformer;remote sensing (RS) scene classification;zero-shot learning (ZSL)},
  doi={10.1109/JSTARS.2024.3414499}}
```
