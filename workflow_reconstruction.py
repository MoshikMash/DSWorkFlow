import ast
import json
import os
import pandas as pd
import datetime
import pathlib
import numpy as np

from copy import deepcopy


def get_file_numbers(file_names):
    file_numbers = []
    for f in file_names:
        second_underscore_index = [i for i, x in enumerate(f) if x == '_'][-1]
        period_index = f.index('.')
        file_numbers.append(int(f[second_underscore_index + 1: period_index]))
    file_numbers.sort()
    return file_numbers


def get_created_time(file_name):
    file_name = pathlib.Path(file_name)
    return datetime.datetime.fromtimestamp(file_name.stat().st_mtime)


class WorkFlowReconstruction:
    def __init__(self, path, session_name):
        self.df = None
        self.session_name = session_name
        self.path = path

    def get_time_from_start_in_seconds(self, current_time):
        difference = current_time - self.df['time'].iloc[0]
        seconds_in_day = 24 * 60 * 60
        return difference.days * seconds_in_day + difference.seconds

    @staticmethod
    def get_time_from_start_minutes_and_seconds(time_from_start_in_seconds):
        return str(divmod(time_from_start_in_seconds, 60))

    def drop_bug_moments(self, start_bug_time_sec, end_bug_time_sec):
        rows_to_drop = []
        for index, row in self.df.iterrows():
            if start_bug_time_sec <= row['time_from_start_sec'] <= end_bug_time_sec:
                rows_to_drop.append(index)
        self.df.drop(index=rows_to_drop, inplace=True)

        bug_length = end_bug_time_sec - start_bug_time_sec
        self.df['time_from_start_sec'] = np.where(self.df['time_from_start_sec'] >= end_bug_time_sec,
                                                  self.df['time_from_start_sec'] - bug_length,
                                                  self.df['time_from_start_sec'])

    def init_executed_cells_list_if_needed(self, file_name, executed_cells):
        """
        add conditions for experimental sessions that contain 2 or more folders.
        """
        if '1866' in file_name:
            return set()
        else:
            return executed_cells

    def reconstruct_workflow(self):
        executed_cells = set()

        cell_info = []
        cell_num = 0

        file_numbers = get_file_numbers(os.listdir(self.path))
        file_names = [f'{self.session_name}_{num}.ipynb' for num in file_numbers]

        for filename in file_names:
            current_notebook = self.path + '/' + filename

            if not os.path.isdir(current_notebook):
                notebook_text = open(current_notebook, 'r').read()

                if self.session_name == 'Pilot_3':
                    executed_cells = self.init_executed_cells_list_if_needed(filename, executed_cells)

                if notebook_text == '':
                    continue

                notebook = dict(json.loads(notebook_text))
                cells = notebook['cells']

                for cell in cells:
                    if cell['cell_type'] == 'code':
                        cell_num = cell_num + 1
                        exec_num = cell['execution_count']

                        output_msg = str(cell['outputs'])
                        output_text = []
                        output_type = []

                        error = 0
                        error_name = 0

                        if cell['outputs']:
                            for output in cell['outputs']:
                                try:  ## Find any text outputs, add to list
                                    for text in output['text']:
                                        output_text.append(text)
                                        output_type.append(output['output_type'])
                                except:
                                    pass

                                try:  ## Find any text/plain outputs (no print statement)
                                    for text in output['data']['text/plain']:
                                        output_text.append(text)
                                        output_type.append(output['output_type'])
                                except:  ## If there are none, this is an empty list
                                    output_type = []
                                if 'error' in str(output['output_type']).lower():
                                    ## Find if there is an error, and the error type
                                    error = 1
                                    error_name = output['ename']

                        if exec_num is None:
                            continue

                        if exec_num not in executed_cells:
                            sources = cell['source']
                            for source in sources:
                                ## Loop through each line in this cell
                                if source[0] != '#':
                                    cell_info.append((cell_num, exec_num, current_notebook, notebook_text, source,
                                                      output_type, output_text, error, error_name))
                            executed_cells.add(exec_num)

        self.df = pd.DataFrame(data=cell_info,
                               columns=['cell_num', 'exec_count', 'file_name', 'notebook_content', 'code',
                                        'output_type', 'output_text', 'has_error', 'error_type'])

        self.df['time'] = self.df['file_name'].apply(get_created_time)
        self.df['time_from_start_sec'] = self.df['time'].apply(self.get_time_from_start_in_seconds)
        self.df['time_from_start_sec_original'] = deepcopy(self.df['time_from_start_sec'])
        # add this code if an experimental session includes a bug
        if self.session_name == 'Pilot_1':
            self.drop_bug_moments(start_bug_time_sec=1265, end_bug_time_sec=1795)
        self.df['time_from_start_(M,S)'] = self.df['time_from_start_sec'].apply(
            self.get_time_from_start_minutes_and_seconds)
        self.df['time_from_start_(M,S)_original'] = self.df['time_from_start_sec_original'].apply(
            self.get_time_from_start_minutes_and_seconds)
