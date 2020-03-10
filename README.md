# Complex Systems Thesis - "Lab book"

![alt text](https://www.floris.cc/shop/675-home_default/momentary-push-button-switch-12mm-square.jpg)

## Purpose of the repo:
  - Keeping track of evolution of the code over time, collect bibliography and notes ideas/issues

## How to use the code:
  - "bistable_switch" replicates Kryven et al. "Solution of the chemical master equation by radial basis functions approximation with interface tracking" for bistable switch case.
    Comments in the python codes are self explanatory. Note that no "for" loop was used to improve time performances, most of the time in the computation is in i/o process. A smarter output for data and .gif must be considered. 
  
  - Folders with results data/outputs will automatically created/updated when running the codes.
  
### Ideas:
  - Interpreting evolution of the systems from propensities map
  - Implement edge detection and keep track of support displacements for given values of PDF ratio over total PDF
  
### Issues:
  - Output files are now sampled at every time step and are >4Gb (space* space * time = 300x300x5000): consider writing output data 1:10(0) and fix this in the plot functions ["solved"!]
