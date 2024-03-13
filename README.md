# CPPN Squared
- Using a CPPN (CPPN Squared) to predict the weights of another CPPN (CPPN1), then plug the predicted weights back in the first one to see what comes out
- this is the kind of cursed project that happens when an art student does machine learning

## Breakdown of files
#### ipynb notebooks
- [CPPN_squared1.ipynb](https://github.com/iuncus/CPPNsquared/blob/main/CPPN_squared1.ipynb) is the current baseline of this project, developed from CPPN_squared.ipynb
- [CPPN_squared.ipynb](https://github.com/iuncus/CPPNsquared/blob/main/CPPN_squared.ipynb) was the first iteration of the code, and has since been deprecated
- [CPPN_squared1 loss test.ipynb](https://github.com/iuncus/CPPNsquared/blob/main/CPPN_squared1%20loss%20test.ipynb) is a fork of CPPN_squared1.ipynb, where I tried to calculate the loss based on the output image compared to im_000078.png
- [training_loopception.ipynb](https://github.com/iuncus/CPPNsquared/blob/main/training_loopception.ipynb) is another fork of CPPN_squared1.ipynb, where I tried to train a different instance of CPPN1 in every training step of CPPN Squared
- [CPPN_squared1 bisected.ipynb](https://github.com/iuncus/CPPNsquared/blob/main/CPPN_squared1%20bisected.ipynb) is yet another fork of CPPN_squared1.ipynb, where I tried to train CPPN Squared to recreate only one layer in CPPN1
- [Output_test.ipynb](https://github.com/iuncus/CPPNsquared/blob/main/Output_test.ipynb) is used to load a checkpoint into CPPN1 and predict an image
#### folders
- [Chekpoints](https://github.com/iuncus/CPPNsquared/tree/main/Checkpoints) folder contains all the relevant checkpoints, either loaded for training or the output of said training
- [Interesting_models](https://github.com/iuncus/CPPNsquared/tree/main/Interesting_models) folder contains some models that produce interesting results when fed through Output_test.ipynb
- [src](https://github.com/iuncus/CPPNsquared/tree/main/src) folder contains 2 python files, one for normalization utilities and the other contains the entirety of CPPN1
#### images
- [ACNMW_ACNMW_DA000182-001.jpg](https://github.com/iuncus/CPPNsquared/blob/main/ACNMW_ACNMW_DA000182-001.jpg) is the image used to train CPPN1
- [im_000078.png](https://github.com/iuncus/CPPNsquared/blob/main/im_000078.png) is the prediction output of CPPN1

  
