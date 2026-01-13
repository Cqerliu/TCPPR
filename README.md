TCPPR: A Promoter Prediction Model with Fusion of RNA Polymerase Sequence Information

Description of the document:
The data folder contains promoter and RNAP sequence data for all species.Among them, AT refers to Arabidopsis thaliana, BS refers to Bacillus subtilis, coli refers to Escherichia coli, HM refers to Homo sapiens, MM refers to Mus musculus, and SC refers to Saccharomyces cerevisiae.
The encode.py file is used to encode two types of sequences.
The TCPPR_module.py file defines the feature extraction module, feature fusion module, and MLP.
TCPPR.py is designed for performing 5-fold cross-validation on species promoters, requiring two types of sequences as input.
The visualization.py script defines functions for plotting confusion matrices and feature contribution comparison charts.

Instructions for use:
After inputting the raw data of the two sequences, they are first shuffled synchronously and randomly, then split via five-fold cross-validation.
Subsequently, the training set and test set are respectively subjected to encoding, feature extraction, and feature fusion operations, and finally the training and testing of promoters are completed.
Therefore, only by modifying the reading path of data files in the TCPPR.py file can the prediction for different species be realized.
If you want to perform prediction using only promoter sequences, comment out the RNAP-related parts during training and testing, change the loss function to standard cross-entropy, remove the learning rate annealing, and switch the optimizer to Adam.