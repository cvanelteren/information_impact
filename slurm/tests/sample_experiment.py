from slurm.Task import Task


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
        return {"hello there" : "how do you do?"}
