# CPPN Squared

## Overview
CPPN Squared is an experimental project that leverages a [Compositional Pattern-Producing Network (CPPN)](https://en.wikipedia.org/wiki/Compositional_pattern-producing_network) to predict the weights of another CPPN. The predicted weights are then fed back into the original network to analyze the output. This is the kind of project that happens when an art student gets their hands on machine learning.


## Repository Structure

### Jupyter Notebooks
- **[CPPN_squared1.ipynb](https://github.com/iuncus/CPPNsquared/blob/main/CPPN_squared1.ipynb)** – The current baseline of the project, refined from earlier iterations.
- **[CPPN_squared.ipynb](https://github.com/iuncus/CPPNsquared/blob/main/CPPN_squared.ipynb)** – The initial version of the project, now deprecated.
- **[CPPN_squared1_loss_test.ipynb](https://github.com/iuncus/CPPNsquared/blob/main/CPPN_squared1%20loss%20test.ipynb)** – A fork of `CPPN_squared1.ipynb`, where loss is computed based on image output compared to `im_000078.png`.
- **[training_loopception.ipynb](https://github.com/iuncus/CPPNsquared/blob/main/training_loopception.ipynb)** – A variation that trains a different instance of CPPN1 at each training step.
- **[CPPN_squared1_bisected.ipynb](https://github.com/iuncus/CPPNsquared/blob/main/CPPN_squared1%20bisected.ipynb)** – Another experimental fork where training is focused on recreating only one layer of CPPN1.
- **[Output_test.ipynb](https://github.com/iuncus/CPPNsquared/blob/main/Output_test.ipynb)** – A utility notebook for loading checkpoints into CPPN1 and generating images.

### Key Directories
- **[Checkpoints](https://github.com/iuncus/CPPNsquared/tree/main/Checkpoints)** – Contains model checkpoints for training and inference.
- **[Interesting_models](https://github.com/iuncus/CPPNsquared/tree/main/Interesting_models)** – Includes models that produce notable or unique outputs when processed.
- **[src](https://github.com/iuncus/CPPNsquared/tree/main/src)** – Contains core Python scripts:
  - `CPPN1.py`: Implements the base CPPN model.
  - `util.py`: Provides utility functions for normalization and preprocessing.

### Images
- **[ACNMW_ACNMW_DA000182-001.jpg](https://github.com/iuncus/CPPNsquared/blob/main/ACNMW_ACNMW_DA000182-001.jpg)** – Used for training CPPN1.
- **[im_000078.png](https://github.com/iuncus/CPPNsquared/blob/main/im_000078.png)** – Output prediction from CPPN1.

## Usage

Open one of the provided `.ipynb` files to explore the project.

