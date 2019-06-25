import os, unittest


print(os.curdir)
tests = set()
for root, dirs, fileNames in os.walk(os.curdir):
    for fileName in fileNames:
        try:
            module = fileName.split('.')[0]
            if 'test_' in module:
                tests.add(module)
        except Exception as e:
            print(e)
if __name__ == '__main__':
    print(tests)
    testSuite = unittest.TestSuite()
    tests = unittest.defaultTestLoader.loadTestsFromNames(tests)
    testSuite.addTests(tests)
    unittest.TextTestRunner().run(testSuite)

