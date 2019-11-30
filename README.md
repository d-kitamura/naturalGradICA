# Independent Component Analysis Based on Natural Gradient Algorithm

## About
Sample MATLAB script for independent component analysis (ICA) based on natural gradient algorithm and its application to blind audio source separation.

## Contents
- input [dir]:      includes test audio signals (dry source signals)
- main.m:           main script with parameter settings
- naturalGradIca.m:	function of ICA based on natural gradient algorithm

## Usage Note
ICA assumes instantaneous mixing system (mixing matrix applied to time-domain signal) and separate sources by estimating inverse matrix. This mixing assumption is invalid in an actual audio mixing case because room reverberation exists anywhere. In such a reverberant mixing situation, the mixing system becomes not the instantaneous mixture but convolutive mixture. Therefore, simple ICA cannot separate such reverberant audio mixtures. In this sample script, the source signals are mixed with instantaneous mixture matrix A, and ICA estimates its inverse matrix in a blind manner.

## Original paper
ICA and its optimization algorithm (natural gradient) were proposed in the following papers, respectively:
* P. Comon, "Independent component analysis, a new concept?," Signal processing, vol. 36, no. 3, pp. 287-314, 1994.
* S. Amari, "Natural gradient works efficiently in learning," Neural Computation, vol. 10, no. 2, pp. 251-276, 1998.
## See Also
* HP: http://d-kitamura.net