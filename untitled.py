from pathos.multiprocessing import ProcessingPool as Pool
def calculate(x):
   def domath(y):
       return x*y

   return Pool(4).map(domath, range(3000000))
calculate(2)
