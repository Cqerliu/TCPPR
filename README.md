TCPPR: A Promoter Prediction Model with Fusion of RNA Polymerase Sequence Information

Description of the document:
The data folder contains promoter and RNAP sequence data for all species.
The encode.py file is used to encode two types of sequences.
The TCPPR_module.py file defines the feature extraction module, feature fusion module, and MLP.
TCPPR.py is designed for performing 5-fold cross-validation on species promoters, requiring two types of sequences as input.

Instructions for use:
After inputting the raw data of the two sequences, they are first shuffled synchronously and randomly, then split via five-fold cross-validation.
Subsequently, the training set and test set are respectively subjected to encoding, feature extraction, and feature fusion operations, and finally the training and testing of promoters are completed.
Therefore, only by modifying the reading path of data files in the TCPPR.py file can the prediction for different species be realized.