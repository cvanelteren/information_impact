import sys, unittest, re

sys.path.insert(0, '../')
from Utils import IO, misc
PATH = '../Data/new6/2019-06-06T17:58:42.299717'
class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.path = PATH # change this
    def test_loader(self):
        fileNames = misc.flattenDict(IO.DataLoader(self.path))
        pattern = '((\d+-\d+=\d+T\d+:\d+:)?(\d+\.\d+))'
        print(fileNames[:5])
        for first, second in zip(fileNames[:-1], fileNames[1:]):
            self.assertGreater(second, first)

if __name__ == '__main__':
    unittest.main()
