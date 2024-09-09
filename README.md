# CYPHER Training School - Machine Learning Methods for Reacting Flows (09/09/2024-11/09/2024)

This repository contains the hands-on jupyter notebooks for the Reduced-Order Modelling II sessions of the CYPHER Training School "Machine Learning Methods for Reacting Flows".

This repository contains:

1. [The slides](CYPHER_ML_Course.pdf) presented during the session.
2. The various datasets used during the session
	- The [Lorenz system dataset](Lorenz_data/LorenzSys_Lorenz_data.npz) for the example notebook
	- The [broadband forcing and heat release rate signals](Flame_data/) for the flame dynamics learning exercise
	- The [validation flame describing functions](Validation_data/Flame_valid_data.h5) containing the frequency response of the flam
	- The above data is also provided in a combined [zip](CYPHER_SCHOOL_data.zip) for convenience

## Description
The repository contains three different notebooks:

1. The [MLP-based Lorenz system notebook](01_MLP_Lorenz.ipynb) where a feedforward neural network is developed to learn the dynamics of the Lorenz system.
2. The [RNN-based Lorenz system notebook](01_LSTM_Lorenz.ipynb) where a long short-term neural network is developed to learn the dynamics of the Lorenz system.
3. The [flame dynamics notebook](02_FlameDynamics.ipynb) which describes how to read the flame dynamics dataset.

