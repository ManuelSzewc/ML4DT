# ML4DT: Machine Learning for Drug Therapies

A repository for 

## A Machine Learning alternative to placebo-controlled clinical trials upon new diseases: A primer
### Ezequiel Alvarez, Federico Lamagna , Manuel Szewc

*You can run all notebooks in Binder by pressing this button:* [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ManuelSzewc/ML4DT/master)

The script runs on Python, it requires the Keras package for the building and training of the Neural Networks.
We include both the Jupyter notebook version and the plain Python script.
 
In this repository we include the necessary files to reproduce plots present in article /arxiv-id/

The main components are:

-Selecting the "health function" H

-Selecting the number of drug trials and patients for the Regular Drug Therapy (RDT)

-Setting the hyperparameters of the Neural Network (Architecture, number of epochs, learning rate, dropout)

-Setting values for the stochastic noise 

The script then runs through the different architectures, noise values and number of trials, and reproduces the plots. It saves them in a folder of the user's choice, along with a log file where you can see the best performing methods, along with the values of the different tests (Spearman Rank correlation R, mean squared error). 

