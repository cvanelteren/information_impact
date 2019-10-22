import os, datetime

root = '/var/scratch/cveltere/tester'
root = 'Data/test/'
THRESHOLD = datetime.datetime(\
year = 2019,\
day  = 18,\
month = 10,\
hour = 17,\

)
print(THRESHOLD)
if __name__ == "__main__":
    delete = []
    counter = 0
    for (root, dirs, fileNames) in os.walk(root):
        for file in fileNames:
            counter += 1
            path = os.path.join(root, file)
            ctime = os.path.getmtime(path)
            ctime = datetime.datetime.fromtimestamp(ctime)
            # if file.endswith('.bk'):
                # os.rename(path, path.split('.bk')[0])
            print(ctime, THRESHOLD, file)
            if ctime > THRESHOLD and 'settings' not in path:
                delete.append(path)
    for d in delete:
        print(d)
        assert 'settings' not in d
        assert not d.endswith('.bk')
        os.rename(d, d + '.bk')
    print(len(delete), counter)
