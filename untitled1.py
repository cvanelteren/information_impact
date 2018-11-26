fileName = 'test.txt'
with open(fileName, 'r') as f:
    unique = set()
    for line in f.readlines():
        unique = unique | set(line.split()[:2])
print(unique)
    