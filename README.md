# Complex Systems Thesis 

![alt text](https://www.floris.cc/shop/675-home_default/momentary-push-button-switch-12mm-square.jpg)

## Summary
  As a final step for the ** Complex Systems ** profile in the MSc in Experimental Physics at Utrecht University, we are required to work on a 15EC project. This project must have two supervisors from two different departments (other from physics). This repo in particular is meant to keep track of evolution of the project over time, to collect bibliography, notes, ideas and issues.

## Outline of the project

### Main steps
  Simulate

#### How to use the code:
  - "bistable_switch" replicates Kryven et al. "Solution of the chemical master equation by radial basis functions approximation with interface tracking" for bistable switch case. Comments in the python codes are self explanatory. Note that no "for" loop was used to improve time performances.
  - Julia version of the code is meant to replicate results from .py but in a way faster way.
  - Folders with results data/outputs will automatically created/updated when running the codes [PY].
  
#### Ideas:
  - Interpreting evolution of the systems from propensities map
  - Implement edge detection and keep track of support displacements for given values of PDF ratio over total PDF
  - Multithread on Julia!
  
#### Issues:
  - Output files are now sampled at every time step and are >4Gb (space* space * time = 300x300x5000): consider writing output data 1:10(0) and fix this in the plot functions [*"solved"*!]
