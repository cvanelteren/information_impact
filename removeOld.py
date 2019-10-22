import os, datetime

root = '/var/scratch/cveltere/tester'
THRESHOLD = datetime.datetime(\
year = 2019,\
day  = 18,\
month = 10,\
hour = 17,\

)
print(THRESHOLD)
if __name__ == "__main__":
    delete = []
    for (root, dirs, fileNames) in os.walk(root):
        for file in fileNames:
            path = os.path.join(root, file)
            ctime = os.path.getctime(path)
            ctime = datetime.strptime(ctime,\
            '%Y-%m-%d %H:%M:%S')
            if ctime > THRESHOLD:
                delete.append(path)
    print(len(delete), delete[:5])
