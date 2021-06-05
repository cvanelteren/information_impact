import sys; sys.path.insert(0, '../')
from Task import Task

import time
def setup(config: dict) -> list:
    tasks = []
    for i in range(1):
        tasks.append(Experiment(config))
    return tasks

class Experiment(Task):
    def __init__(self, settings = {}, output_directory = "test_directory"):
        super(Experiment, self).__init__(settings = settings,
                                   output_directory = output_directory)

    def run(self):
        time.sleep(3)
        return {"hello there" : "how do you do?"}
