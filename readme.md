Table of contents:

- Wasserstein accuracy

- Results:

> source = (60, 50, 140, 50)

> source = (110, 50, 150, 50)

> source = (110, 10, 150, 10)

> triangles and circles, source = (60,50,140,50)

> triangles and circles, source = (110,50,150,50)


# Wasserstein accuracy

Given the FID function from PyTorch described [here](https://pytorch.org/ignite/generated/ignite.metrics.FID.html), I implemented the Wasserstein as in [here](https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py#L152). My implementation is essentially a copy-paste and can be found under [functions.py](https://github.com/MarcoFurlan99/7_Wasserstein_computation_and_more/blob/master/functions.py).

In this experiment I generated $m$ samples from two $n$-dimensional Normal Gaussian distributions $N(0,I_n), N(0,I_n)$.

- x-axis is $n \in [1,99]$.

- y-axis is Wasserstein (should be 0).

- legend is $m \in \lbrace 10^3,10^4,10^5 \rbrace $.

<img src="https://github.com/MarcoFurlan99/7_Wasserstein_computation_and_more/blob/master/images/Wasserstein_accuracy.png?raw=true">

I fit some polynomials and found out that the following:

$$ f(m,n) = \frac{1}{m} (0.502 n^2 + 2.596 n - 4.457) $$

fits perfectly all 3:

<img src="https://github.com/MarcoFurlan99/7_Wasserstein_computation_and_more/blob/master/images/Wasserstein_accuracy_polyy.png?raw=true">

So the polynomial $f$ can be used as a predictor for the Wasserstein error.

In the case of the last latent space of a U-Net, so far I used 1000 images, which totals in 1000x4x4 = 16000 samples in a 1024-dimensional distribution. Setting $m = 16000$ and $n=1024$ in the polynomial gives:

$$ f(16000,1024) = \frac{1}{16000} (0.502 \times 1024^2 + 2.596 \times 1024 - 4.457) \approx 33 $$

I verified that indeed if I generate 16000 samples from two random Standard Gaussian distributions $N(0,I_n)$ the computed value is 32.9022 .

So the error size of the Wasserstein, in the simplest case possible of two standard random Gaussian distributions, is $\approx 33$ with our example. 

In the case $n = 1024$ we have approximately $f(1,1024) = 529039$, consequently to get an error size $f(m,1024) \leq \varepsilon$ we need $m \geq \frac{529039}{\varepsilon}$, let us pose $m = N \times 4 \times 4$ where $N$ is the number of images, then we need $N \geq \frac{33065}{\varepsilon}$.

For $\varepsilon = 1$ we get $N \geq 33065$.
For $\varepsilon = 0.1$ we get $N \geq 330650$.

These numbers are a little ridicolous to work with, we cannot expect to have more than 10.000 images on our target dataset.

The issue is sparsity caused by high dimensionality, so reducing the dimensionality would be a fix.

Based on these results, we can draw a simple graph which tells us the reliability of the Wasserstein calculated on the different latent spaces, as shown below:

<img src="https://github.com/MarcoFurlan99/7_Wasserstein_computation_and_more/blob/master/images/UNet_ls.png?raw=true">

Given a target dataset with 1000 images, these are the data:

|                  | ls0             | ls1            | ls2            | ls3            | ls4            |
| ---------------- | --------------- | -------------- | -------------- | -------------- | -------------- |
| $W \times H$     | $64 \times 64$  | $32 \times 32$ | $16 \times 16$ | $8 \times 8$   | $4 \times 4$   |
| $m$ (n_samples)  | $4096000$       | $1024000$      | $256000$       | $64000$        | $16000$        |
| $n$ (n_channels) | $64$            | $128$          | $256$          | $512$          | $1024$         |
| $f(m,n)$         | $<10^{-3}$      | $<10^{-3}$     | $0.13$         | $2.08$         | $33$           |

Last row is the Wasserstein estimated accuracy. This means that the computation of the Wasserstein is less reliable as we go deeper through the network, because of the lower number of samples and higher dimensionality (increasing sparsity), especially for lower values of the Wasserstein. This may also explain the presence of negative values in the future graphs (ideally $w \geq 0 \implies 1+\log(w) \geq 0$, where w is going to be the Wasserstein distance).

# Results

## source = (60, 50, 140, 50)

Results (before BN adaptation, after BN adaptation, difference):

<img src="https://github.com/MarcoFurlan99/7_Wasserstein_computation_and_more/blob/master/(60,50,140,50)/three_musketeers.png?raw=true">
 
Training history (10 epochs --> 50 validation rounds, early stopping patience = 10):

<img src="https://github.com/MarcoFurlan99/7_Wasserstein_computation_and_more/blob/master/(60,50,140,50)/training_history.png?raw=true">

Wasserstein:

<img src="https://github.com/MarcoFurlan99/7_Wasserstein_computation_and_more/blob/master/(60,50,140,50)/wasserstein_0.png?raw=true">

<img src="https://github.com/MarcoFurlan99/7_Wasserstein_computation_and_more/blob/master/(60,50,140,50)/wasserstein_1.png?raw=true">

<img src="https://github.com/MarcoFurlan99/7_Wasserstein_computation_and_more/blob/master/(60,50,140,50)/wasserstein_2.png?raw=true">


<img src="https://github.com/MarcoFurlan99/7_Wasserstein_computation_and_more/blob/master/(60,50,140,50)/wasserstein_3.png?raw=true">


<img src="https://github.com/MarcoFurlan99/7_Wasserstein_computation_and_more/blob/master/(60,50,140,50)/wasserstein_4.png?raw=true">

log Wasserstein vs Adapted IoU:

<img src="https://github.com/MarcoFurlan99/7_Wasserstein_computation_and_more/blob/master/(60,50,140,50)/Prometheus_0.png?raw=true">

<img src="https://github.com/MarcoFurlan99/7_Wasserstein_computation_and_more/blob/master/(60,50,140,50)/Prometheus_1.png?raw=true">

<img src="https://github.com/MarcoFurlan99/7_Wasserstein_computation_and_more/blob/master/(60,50,140,50)/Prometheus_2.png?raw=true">

<img src="https://github.com/MarcoFurlan99/7_Wasserstein_computation_and_more/blob/master/(60,50,140,50)/Prometheus_3.png?raw=true">

<img src="https://github.com/MarcoFurlan99/7_Wasserstein_computation_and_more/blob/master/(60,50,140,50)/Prometheus_4.png?raw=true">

Unfortunately there seems to be no linear relationship! One fact that may be relevant is that if the Wasserstein in the first layers is low ($<3$), then the adapted_IoU is (very likely to be) high ($>0.9$). This may be obvious if the target datasets are very close to the source one (so the IoU is high regardless of BN adaptation), I'll need to check if that's the case.

From the deeper layers (ls3, ls4) I wouldn't get any conclusion given the complete absence of pattern, which is likely caused by the instability in the computations of the Wasserstein.

The Target-Normalized Wasserstein distance may be a viable option for the first layers. I'll proceed with some testing and see how stable it is.

## source = (110, 50, 150, 50)

Results (before BN adaptation, after BN adaptation, difference):

<img src="https://github.com/MarcoFurlan99/7_Wasserstein_computation_and_more/blob/master/(110,50,150,50)/three_musketeers.png?raw=true">
 
Training history (10 epochs --> 50 validation rounds, early stopping patience = 10):

<img src="https://github.com/MarcoFurlan99/7_Wasserstein_computation_and_more/blob/master/(110,50,150,50)/training_history.png?raw=true">


## source = (110, 10, 150, 10)

Results (before BN adaptation, after BN adaptation, difference):

<img src="https://github.com/MarcoFurlan99/7_Wasserstein_computation_and_more/blob/master/(110,10,150,10)/three_musketeers.png?raw=true">
 
Training history (10 epochs --> 50 validation rounds, early stopping patience = 10):

<img src="https://github.com/MarcoFurlan99/7_Wasserstein_computation_and_more/blob/master/(110,10,150,10)/training_history.png?raw=true">


## triangles and circles, source = (60,50,140,50)

Identify **circles**

<img src="https://github.com/MarcoFurlan99/7_Wasserstein_computation_and_more/blob/master/(60,50,140,50)_toc/samples.png?raw=true">

Results (before BN adaptation, after BN adaptation, difference)

<img src="https://github.com/MarcoFurlan99/7_Wasserstein_computation_and_more/blob/master/(60,50,140,50)_toc/three_musketeers.png?raw=true">


Training history (50 epochs --> 250 validation rounds, early stopping patience = 20):

*lost it rip*


## triangles and circles, source = (110,50,150,50)

Identify **circles**

<img src="https://github.com/MarcoFurlan99/7_Wasserstein_computation_and_more/blob/master/(110,50,150,50)_toc/samples.png?raw=true">

Results (before BN adaptation, after BN adaptation, difference)

<img src="https://github.com/MarcoFurlan99/7_Wasserstein_computation_and_more/blob/master/(110,50,150,50)_toc/three_musketeers.png?raw=true">


Training history (50 epochs --> 250 validation rounds, early stopping patience = 20):

<img src="https://github.com/MarcoFurlan99/7_Wasserstein_computation_and_more/blob/master/(110,50,150,50)_toc/training_history.png?raw=true">