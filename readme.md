# Wasserstein accuracy

In this experiment I generated $m$ samples from two $n$-dimensional Normal Gaussian distributions $N(0,I_n)$.

x-axis is $n \in [1,99]$.

y-axis is Wasserstein (should be 0).

legend is $m \in \{ 10^3,10^4,10^5 \} $.

<img src="https://github.com/MarcoFurlan99/7_Wasserstein_computation_and_more/blob/master/images/Wasserstein_accuracy.png?raw=true">

I fit the polynomials and found out that the following polynomial:

$$ f(m,n) = \frac{1}{m} (0.502 n^2 + 2.596 n - 4.457) $$

fits perfectly all 3:

<img src="https://github.com/MarcoFurlan99/7_Wasserstein_computation_and_more/blob/master/images/Wasserstein_accuracy_poly.png?raw=true">

