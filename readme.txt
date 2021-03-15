Files: 
⁃	save_history.py
⁃	action_tagging.py
⁃	researchMapping.py

Save Jupyter notebook sanpshots:
- Run save_history.py
- You should modify the paths at the begining of the file in order to determine which notebook should be tracked and the destination location for the notebook snapshots. 

Code reconstruction:
⁃	Place  action_tagging.py  into the folder with the Jupyter notebook files. 
⁃	The structure of these files is assumed to be P*/P*_task*/experiment/*****.ipynb
⁃	on the command line, navigate to the folder containing action_tagging.py and the participant folders.
⁃	Run this file for each person and task with:
python action_tagging.py "P1/P1_task2/experiment" "P1T2.p"
⁃	The new parsed pickle will be stored in “P1T2.p" 
⁃	The participant is 1, and task number is 2 (in this example)

Action assigment and feature extraction:
⁃	Move parsed pickles to folder containing researchMapping.py 
⁃	For each Participant and Task: Change first and second input for mapExecutions to the place where the parsed pickle is, and the output file name/location you want to save the pickle to
⁃	Run mapExecutions 

⁃	csv files will be saved to directly csv/
⁃	Pickle files will be saved to directory pickles/ 
