class Test:
    def __init__(self, x, y):
        for i in locals().copy():
            if i != 'self':
                exec('self.{} = {}'.format(i,i))
                print(i)
t = Test(1,1)
print(t.x)
