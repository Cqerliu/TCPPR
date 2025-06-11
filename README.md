# TCPPR
TCPPR: Deep fusion of RNA polymerase sequence information for promoter prediction model

## Description of the document:
The data folder contains promoter and RNAP sequence data for all species.
The data_process folder holds code for feature extraction.
The fused_features folder is used to store fused feature vectors.
TCPPR.py is designed for performing 5-fold cross-validation on species promoters, requiring only fused vectors as input.

## Instructions for use:
In the data_process folder, ft.py is used to extract promoter features, and another ft.py (note: ensure feature dimensions match those of promoters) extracts RNAP features.
Fusion.py fuses the two types of features. If the paths for FT (promoter features) and FC (RNAP features) are identical, it indicates no fusion of RNAP features; different paths mean fused feature vectors.
