# Multi-scale-Multimodal-Cross-Attention-for-Seizure-Detection
#### Model 1 features 
* CNN-Multiscale multimodalities
* weighted sum cls token fusion on long and short data 
* weighted sum original embedding and transformed embedding on long and short data 
* cross-attention is done on the new cls token and the new embeddings

#### Model 1 features 
* CNN-Multiscale multimodalities
* sum of cls tokens for on long and short data
* cls tokens added back to transformed embedding for long and short data
* cross-attention is done on fused transformed embedding for long and short data separately
