# Neural Speech Phase Prediction based on Parallel Estimation Architecture and Anti-Wrapping Losses
### Yang Ai, Zhen-Hua Ling

In our [paper](https://arxiv.org/xxx), 
we proposed a novel speech phase prediction model which predicts wrapped phase spectra directly from amplitude spectra by neural networks.<br/>
We provide our implementation and pretrained models as open source in this repository.

**Abstract :**
This paper presents a novel speech phase prediction model which predicts wrapped phase spectra directly from amplitude spectra by neural networks. The proposed model is a cascade of a residual convolutional network and a parallel estimation architecture. The parallel estimation architecture is composed of two parallel linear convolutional layers and a phase calculation formula, imitating the process of calculating the phase spectra from the real and imaginary parts of complex spectra and strictly restricting the predicted phase values to the principal value interval. To avoid the error expansion issue caused by phase wrapping, we design anti-wrapping training losses defined between the predicted phase spectra and natural ones by activating the instantaneous phase error, group delay error and instantaneous angular frequency error using an anti-wrapping function, respectively. Experimental results show that our proposed neural speech phase prediction model outperforms the iterative Griffin-Lim algorithm and other neural network-based method, in terms of both reconstructed speech quality and generation speed.

Visit our [demo website](http://staff.ustc.edu.cn/~yangai/NSPP/demo.html) for audio samples.
