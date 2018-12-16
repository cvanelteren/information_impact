import IO, os, plotting

workingDir = os.getcwd()
dataPath   = f'{workDir}/Data/'
figuresPath= 'f{workDir}/figures/'
if not os.path.exists(figuresPath):
    os.mkdir(figuresPath)

# fitting params
theta      = 1e-2
METHOD     = 'linearmixing'
func       = lambda x, a, b, c, d, e, f: a + b * exp(-c* x) # d * exp(- e * (x - f))
