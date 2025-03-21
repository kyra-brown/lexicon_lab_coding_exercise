import subprocess
import os
import pandas as pd

forager_folder = '/Users/kbrown3/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/GitHub/lexicon_lab_coding_exercise/forager'
data = '/Users/kbrown3/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/GitHub/lexicon_lab_coding_exercise/data-cochlear.txt'
pipeline = 'switches'
switch_method = 'simdrop'
model = 'static'
domain = 'animals'
output_directory = '/Users/kbrown3/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/GitHub/lexicon_lab_coding_exercise/results'


os.chdir(forager_folder)

command = ['python3', 'run_foraging.py', 
           '--data', data, '--pipeline', pipeline, '--switch',
           switch_method, '--model', model, '--domain', domain]

result = subprocess.run(command, check = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE, text = True)
