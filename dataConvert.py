from Utils import IO
import os
root = 'someRoot'
data = IO.DataLoader(root)
data = {os.path.join(root, k): v for k, v in data.items()}

settings = IO.loadSets(data)

from datetime import datetime
stamp = datetime.now().isoformat() + f'_N={len(data)}'
IO.savePickle(stamp, settings)
