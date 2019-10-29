import os

root = 'Data/tester'


for (root, dirs, files) in os.walk(root):
    for file in files:
        if file.endswith('.bk') and 'settings' not in file:
           path = os.path.join(root, file)
           os.remove(path)
