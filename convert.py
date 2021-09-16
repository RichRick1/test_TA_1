import os

path = os.getcwd()
fname='finding_root.py'

os.system('jupyter nbconvert --to script finding_root.ipynb')

with open(fname, 'r') as f:
    lines = f.readlines()
with open(os.path.join(path, 'problem', fname), 'w') as f:
    for line in lines:
        if 'nbconvert --to script' in line:
            break
        else:
            f.write(line)
os.remove(fname)
