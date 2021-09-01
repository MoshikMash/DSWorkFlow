# DSWorkFLow
DSWorkflow is a framework for conducting experiments which investigate data scientists’ thinking processes and which carry out both quantitative as well as qualitative analyses of their workflow.

## DSWorkFLow Components:
1. Data Collection
2. Workflow Reconstruction
3. Feature Extraction

See the following for descriptions of the sub-components which make up each component:

### 1. Data Collection:
A Python script which saves a snapshot of the Jupyter Notebook that the participant is working on at 1-second intervals. 
FIles: data_collection_save_notebook_snapshots.py
data_collection_dsworkflow_api (optional): a Python file with all the pre-developed functions which could assist the participants in completing their assigned task (developing a Machine Learning model).

FIles:
Data_collection_dsworkflow_api.py
DSWorkFLowAPI.docx: DSWorkFlow API function descriptions for experiment participants
Qualitative sub-components: External components used for saving additional information which could be synchronized with the other components with respect to specific timestamps. An example could be Screen, Video and Audio recordings, (recommended software to this end: OBS). The use of screen recordings which include the system clock may also aid the synchronization of this data with other datasets. Another example would be the use of eye tracking equipment.

### 2. Workflow Reconstruction:
A Python Class for reconstructing the workflow in chronological order and extracting general information. This Class contains a variable named df, which is a Pandas dataframe containing the reconstructed workflow.

FIles: workflow_reconstruction.py
	
### 3. Feature Extraction:
A Python class which extracts relevant features from the reconstructed object (reconstructed workflow) for the purposes of analyzing the code workflow.

FIles:
Feature_extraction.py: the class itself
Features Directory: contains Python Transformers for extracting the desired features
Old features Directory: Contains many features that were extracted and not used as they were not relevant to the specific model in question (such as predicting moments of stuckness)
configuration.json: this file includes configuration settings that are used by the feature files (e.g. window sizes, chosen features)

## Additional information can be found in the following paper:
Moshe Mash, Stephanie Rosenthal, and Reid Simmons. 2021. DSWorkFlow: A Framework for Capturing Data Scientists’ Workflows. Extended Abstracts of the 2021 CHI Conference on Human Factors in Computing Systems. Association for Computing Machinery, New York, NY, USA, Article 305, 1–7. DOI:https://doi.org/10.1145/3411763.3451683
 




