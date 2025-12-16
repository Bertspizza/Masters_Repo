# Conflict Strikes Back: The Role of Interpersonal Coordination in Romantic Couples (Masters Thesis)
This repository contains all the relevent code used for the analysis of data present in the thesis. To execute the python scripts, ensure that packages listed in requirements.txt are installed. 
The available data can be installed through the following link: ([Link](https://osf.io/v9mjh/overview))
Due to storage restictions, the raw OpenPose files are only available through making a request.

Cross Recurrence Analysis (CRQA) includes C++ extensions that need to be compiled before use. To compile the crqa relevant code, ensure you have a C++ compiler installed. Then, run the following from the root directory:
- pip install .

## Overview

Directories: 
- iaaft_main: Repository developed by Bedartha Goswami. 
- stats: Contains televant scripts and files for statistical analysis and hypothesis testing.
- utils_dir: Utilities for data processing
- utils_rqa: Utilities for CRQA.

Scripts:
- ami_fnn.ipynb: Used to conduct average mutual information and false nearest neighbours analysis
- check_op_quality.ipynb: Used to assess the quality of the raw pose output data.
- crqa.ipynb: Conduct crqa on the real pairs.
- graph_rqa.ipynb: Graph the crqa output
- Preprocess_openpose_file: Process the raw openpose files (i.e., data cleaning and calculating metrics)
- pseudo_iaaft_crqa.ipynb: CRQA on the iaaft timeseries
- pseudo_pairs_crqa: CRQA on pairs that did not interact

meta data:
- metaData_coding.csv: Contains the frame to trim the video
- partner_role_metadata.csv: Contains the role of the person seated on the right (p1) and left (p2)

## Analysis Steps
Note: Step 1 requires the raw OpenPose data to execute. Additionally, CRQA data is available to use and skip steps 1-4. 'merged_crqa_iaaft.csv' contains the coordination data used for the analysis presented in the thesis.

- Step 1: Process the raw time series data. (Preprocess_openpose_files.ipynb)
- Step 2: Conduct an average mutual analysis and false nearest neighbours analysis on the processed time series. (ami_fnn.ipynb)
- Step 3: Conduct Cross Recurrence Quantification Analysis (CRQA) on the actual pairings. (crqa.ipynb)
    - To plot CRQA outcomes, see the utils_dir/graph_crqa_utils.py 
- Step 3.1: Generate IAAFT time series and perform CRQA. (pseudo_iaaft_crqa.ipynb)
- Step 3.2: Perform CRQA on participants who did not interact. (pseudo_pairs_crqa.ipynb)
- Step 4: Merge actual and baseline outcomes into a single csv (merge_pseudo_actual.ipynb)
- Step 5: Perform statistical analysis (pseudo_comp.ipynb; RQ1.ipynb; RQ2.ipynb; RQ3.ipynb)

## References 

Cross-Recurrence Quantification Analysis (CRQA) was completed using the Recurrence-Quantification-Analysis Github Repository ([link](https://github.com/xkiwilabs/Recurrence-Quantification-Analysis)) (Richardson & Macpherson, n.d.)

Iterative Adjusted Ampliture Fourier Transformation (IAAFT) was completed using the iaaft Repository ([link](https://github.com/mlcs/iaaft)) (Goswami, n.d.)

Pose estimation was competed using the open-sorced OpenPose Model ([link](https://github.com/CMU-Perceptual-Computing-Lab/openpose)) (CMU-Perceptual-Computing-Lab, 2017)
