import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--file')
parser.add_argument('--id')
if __name__ == '__main__':
    args = parser.parse_args()
    
    print(args.file)
    print(args.id)
