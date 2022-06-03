# Improving Redundancy in Attention

## Introduction

This repository contains code for improving the ease at which the attention mechanism in scaled dot product is able to 
redundant (the attention matrix representing an identity matrix). This idea originates from one of the key ideas behind 
residual connections. Residual connections increase the ease at which the identity mapping can be learned, thereby 
reducing unwanted distortions from the operations.

The code is built on top of Swin Transformer V2 and we thank them for their cleanly maintained code base. 

> **SwinV2**: Please refer to [SwinV2_README.md](SwinV2_README.md) for SwinV2, which this code is built on.

> **Setup**: Please refer to [get_started.md](get_started.md) for setting up a CUDA compatible environment and 
> dependencies.

## Additions

### Residual Connections for Q and K

Conventional knowledge states that smaller weight parameters (i.e. from weight decay) results in a simpler model as the 
parameters have a smaller impact and this tends to result in better performing models according to Occam's Razor.

The more similar the attention matrix is to the identity function, the lower impact this operation has on the network. 
Weight decaying Q and K towards zero matrices results in a smoother attention matrix w.r.t. the main diagonal and 
increases the impact this operation has on the network. 

By adding residual connections to Q and K, weight decay Q and K towards X instead. This results in an 
attention matrix more similar to the identity function, and increases the ease at which an identity mapping can be
represented by the attention matrix. 

For compute efficiency, these residual connections can be implemented by adding an identity matrix to the projections 
for Q and K.

### Continuous R.P. weight on Attention Matrix A

We inject a continuous relative position weight (computed similarly to continuous relative position bias) that is 
multiplied with the Attention Matrix. When combined with continuous relative position weight, is able to act as a 
sharpening (/smoothing) function w.r.t. the main diagonal, increasing the ease at which the attention matrix can 
represent the identity matrix.

### Current Results on Original Swin Transformer


|  name   | QK residual | CRPW | Pre MLP | resolution | acc@1 | acc@5 | #params | FLOPs |
|:-------:|:-----------:|:----:|:-------:|:----------:|:-----:|:-----:|:-------:|:-----:|
| Swin-T  |     No      |  No  |   No    |  224x224   | 81.3  | 95.5  |   28M   | 4.5G  |
| Swin-T  |     Yes     |  No  |   No    |  224x224   | 81.4  | 95.5  |   28M   | 4.5G  |
| Swin-T  |     Yes     | Yes  |   No    |  224x224   | 81.6  | 95.6  |   28M   | 4.5G  | 
| Swin-T  |     Yes     | Yes  |   Yes   |  224x224   | 81.8  | 95.8  |   28M   | 4.5G  |



### TODO
- Run experiments for SwinV2
- (Experimental) Pre MLP for each resolution 
- (Experimental) S.W. MLP 