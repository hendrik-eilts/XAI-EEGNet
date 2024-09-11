
# Bridging the Gap: Explainable AI Insights Into EEGNet Classification and Its Alignment to Neural Correlates

## What is it about?

This thesis applies Concept Relevance Propagation (CRP), an eXplainable AI method, to the EEGNet classifier and three EEG datasets (motor imagery, auditory attention, internal vs. external attention). The aim is to establish a link between the concepts learned by the models and the neural correlates of the respective tasks, thereby bridging the gap in explainability between ML and neuroscience. Relevances were computed on individual filters of a convolutional layer, clustering was conducted on these filters to identify common patterns among subjects, and these clusters were further analyzed in the frequency domain using topographic maps. 

Additionally, the relevance computed by CRP was combined with Independdent Component Analysis in a novel approach, providing relevance values for neural and non-neural sources of the EEG signal. Finally, functional grouping was applied to quantify the results. The brain regions and frequency bands identified by the method largely align with the existing literature.

This study represents a step towards a deeper understanding of how DL models decode EEG signals and offers insights into neural correlates in a human-understandable manner.

## Overview

This repository consists of 6 main files:

1. **global_settings.py** <br>
  This file contains the global variables determining which data to use, which models to train and use and how the relevances are computed, as well as the paths to the folders to load and store data from and to. 

2. **main_datasets.py** <br>
  This file handles the computation of epochs-objects from the raw data. Instead of providing the raw datasets, I included the epochs-objects in the cloud instead, so this file doesn't have to be executed. 

3. **main_classification.py** <br>
  This file handles the training of the models of the leave-one-out-cross validation.

4. **main_classification.ipynb** <br>
  This file contains the code used to plot and analyze the results of the classification pipeline.

5. **main_relevance.py** <br>
  This file handles the computation of the relevances (pipeline A) which have to be calculated in order to analyze these using the **main_relevance.ipynb** file. 

6. **main_relevance.ipynb** <br>
  This file contains the code used to plot and analyze the previously computed relevances (pipelines A and B). 

Additionally, the **requirements.txt** file contains the libraries needed to run this code. 


## Tutorial


### 1. Virtual Environment

To run this code, first set up a virtual environment using the libraries defined in requirements.txt. On linux this can be done by entering this line into a terminal: <br>
```python3 -m pip install -r requirements.txt``` <br>
Or on windows:  <br>
```py -m pip install -r requirements.txtt```

More information can be found here: https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#using-a-requirements-file

### 2. Paths and Precomputed Data

The next step is to define the path where you want to load and store data to and from via the the ```base_path```, located in the ```global_settings.py``` file. 

### 3. Process data

You have two options, first you can copy the **processed_data**-folder into the folder specified by ```base_path```, or you can create these files automatically using the ```main_datasets.py``` script. Creating the files new, however, requires the raw data, which I did not include in the cloud. 

### 4. Create classification models

The next step is to either train new models, which are stored under [base_path]/classification_results, or to copy the precomputed models (the folder **classification_results**) from the cloud into your base_path-folder.

If you want to recompute these models for testing purposes, I would advise that you set the test-flag in ```global_settings.py``` to ```True```, which then only includes the internal/external attention dataset, which is the smallest in size.
The computation for this dataset can take upwards of 2 hours.  
The performance of the classifiers can be examined using the file ```main_classification.ipynb```. 

### 5. Compute relevances (pipeline A)

Now, given the classification models, the relevances can be computed using the file ```main_relevance.py```. Again, the precomputed relevances can be found in the cloud in the folder ```results_relevance_0.2_test```. To use these, just copy this folder into the folder of the base_path-folder. However, computing these new for the internal/external attention dataset doesn't take too long (~5min). The word 'test' in the folder refers to the samples that are used to represent filters. In this case, the samples from the test-data have been used. 

### 6. Analyze relevances (pipelne B and C)

Finally, with the computed relevances, the file ```main_relevance.ipynb``` can be executed. This notebook contains the pipelines B and C, as well as other analyses. 
This file creates temporary files in the base_path-folder to prevent unnecessary recomputation of intermediate results. If you don't want to recompute these results, you can copy the files ```results_[dataset]_activation_map_0.2``` into the base_path-folder. 



















