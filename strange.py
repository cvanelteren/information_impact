import os


root = 'Data/tester/'

watchFolders = ['2019-10-16T15:57:15.895559', \
        '2019-10-16T10:54:59.050361',\
        '2019-10-16T10:54:48.274841',\
        '2019-10-16T15:57:27.814220',\
        '2019-10-16T15:57:24.802926',\
        ]
for (root, dirs, paths) in os.walk(root):

    
    for file in paths:
        if file.endswith('.pickle') and 'settings' not in file:
            path = os.path.join(root, file)
            trial = file.split('_')[0]
            trial, number = trial.split('=')
            number = int(number)
    
            if number > 29:
                for test in watchFolders:
                    if test in path:
                        print(path)
                        os.rename(path, path + '.bk')
    
    
