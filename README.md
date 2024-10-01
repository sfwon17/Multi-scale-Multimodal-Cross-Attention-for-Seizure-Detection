# Multi-scale-Multimodal-Cross-Attention-for-Seizure-Detection-with-Biosignals

The paper has been accepted for The 22nd Australasian Data Science and Machine Learning Conference and will be shared once published.

### Overview
The goal was to incorporate multibiosignals for seizure detection, and this study is the first to utilize Transformer-based architecture for this purpose, outperforming previous models. The models developed in this repository aim to enhance the performance demonstrated in the following studies:
1. https://dl.acm.org/doi/fullHtml/10.1145/3373017.3373055
2. https://github.com/sfwon17/Cardiorespiratory-function-as-a-non-invasive-biomarker-for-Detecting-Seizures-with-AI

### Requirements
Then seizure detection transformer models were built on top of models provided at https://github.com/lucidrains/vit-pytorch.

### Dataset
Since there are no publicly available datasets containing multibiosignals alongside EEG for seizure detection, the dataset used to train and evaluate our models consists of data from 64 patients at Royal Melbourne Hospital. This dataset is not publicly available at this time.

### Models 
#### Model 1 features 
* CNN-Multiscale multimodalities
* weighted sum cls token fusion on long and short data 
* weighted sum original embedding and transformed embedding on long and short data 
* cross-attention is done on the new cls token and the new embeddings

#### Model 2 features 
* CNN-Multiscale multimodalities
* sum of cls tokens for on long and short data
* cls tokens added back to transformed embedding for long and short data
* cross-attention is done on fused transformed embedding for long and short data separately
