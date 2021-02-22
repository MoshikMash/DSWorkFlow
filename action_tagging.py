import sys
import json
import os
import numpy as np 
import csv
import pickle

def tag_actions(path):
    taken_actions = set()
    tags = []
    files = [] 
    d = {}
    cell_num = 0
    for filename in os.listdir(path):
        ## Each File for This Task Person Pair
        current_notebook = path + '/' + filename
        if os.path.isdir(current_notebook) == False:
            notebook_text= open(current_notebook, 'r').read()
            if notebook_text == '': continue
            notebook = dict(json.loads(notebook_text))
            cells = notebook['cells']
            for cell in cells:
                ## Each Cell in This File
                if cell['cell_type'] == 'code' and not 'hide_input' in cell['metadata']:
                    cell_num = cell_num + 1
                    #if cell_num == 1086:
                    #    print(cell)
                    exec_num = cell['execution_count']
                    outputmsg = str(cell['outputs'])
                    outputtext= []
                    outputtype = []
                    error = 0
                    error_name = 0
                    if cell['outputs'] != []:
                        ## This cell has outputs
                        for output in cell['outputs']:
                            try: ## Find any text outputs, add to list 
                                for text in output['text']:
                                    outputtext.append(text)
                                    outputtype.append(output['output_type'])
                            except:
                                pass
                            try: ## Find any text/plain outputs (no print statement)
                                for text in output['data']['text/plain']:
                                    outputtext.append(text)
                                    outputtype.append(output['output_type'])
                            except: ## If there are none, this is an empty list
                                outputtype = []
                            if 'error' in str(output['output_type']).lower():
                                ## Find if there is an error, and the error type
                                error = 1
                                error_name = output['ename']
                    if exec_num == None: continue
                    if exec_num not in taken_actions:
                        sources = cell['source']
                        for source in sources:
                            ## Loop through each line in this chunk 
                            if source[0] != '#':
                                tags.append([source, exec_num,cell_num, current_notebook, outputtype,outputtext, error, error_name])
                                files.append( current_notebook)
                        taken_actions.add(exec_num)
    tags = sorted(tags, key = lambda x: int(x[1]))

    return tags, files, d 



if __name__ == '__main__':
    path_to_init_folder = sys.argv[1]
    out, files, d = tag_actions(path_to_init_folder)
    outfile = sys.argv[2]
    for row in out: print(row)
    pickle.dump(out, open(outfile, 'wb'))
