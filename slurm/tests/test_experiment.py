import os
from unittest import Testcase, main
from slurm.Task import Task

SETTINGS = "test_toml.toml"

from run import ExperimentManager
class TestExperimentManager(TestCase):
    """
    A manager should:
    - manage workers
    - dump to disk
    - restart
    """
    def setUp(self):
        self.manager = ExperimentManager(SETTINGS)

    def test_detinstion_directory(self):
        self.assertTrue(os.isdir(self.manager.base))

    # check if it runs
    def test_run(self):
        self.assertEqual(self.manager.run(), None) 

    def test_dump_to_disk(self):

        self.manager.dump_to_disk()
        self.assertTrue(os.exists(self.manager.name + ".pickle"))
    # check restart
    def test_restart(self):
        self.assertEqual(self.manager.run(), None)

    def test_no_created_files(self):
       fp = self.manager.name + ".pickle"
       if not self.assertFalse(os.exists(fp)):
           os.remove(fp)

    # check if no files are left behind
    def tearDown(self):
       self.test_no_created_files()

import sample_experiment
class TestExperiment(TestCase):
    """
    An experiment should
    - setup
    - implement a task
    - run
    """
    def test_setup(self):
        # check if zero input raises
        with self.assertRaises(TypeError):
            sample_experiment.setup()
        # check the output
        tasks = sample_experiment.setup({})

        self.assertEqual(tasks, list)
        # testing magic number (see experiment file)
        self.assertEqual(len(tasks), 1)

        # check if all inputs are Task
        checks = (isinstance(t, sample_experiment.Experiment) for t in tasks) 
        self.assertTrue(all(checks))
        self.tasks = tasks

    def test_run(self):
        results = []
        for t in task:
            r = t.run()
            self.assertEqual(r, {"hello there" : "how do you do"})
        

from slurm.Task import Worker
class TestWorker(TestCase):
    """
    A worker needs to:
    - hold tasks
    - save results
    - restart
    """
    def setUp(self):
        self.tasks = sample_experiment.setup()

        self.worker_settings = dict(deadline = time.time() + 100,
                               id = "test",
                               )

    def test_run(self):
        worker = Worker(self.tasks, **self.worker_settings)
        # should finish
        self.assertEqual(worker.run(), None)
        for t in self.tasks:
            fp = worker.create_output_file(t)
            self.assertTrue(os.exists(fp))
        
    def test_restart(self):
        worker = Worker(self.tasks, **self.worker_settings)
        worker.restart()
        self.assertEqual(worker._running, False)

        worker.clean_up()
        fp = worker.create_dump_name()
        self.assertFalse(os.exists(fp))
    

if __name__ == "__main__":
    main()
        


        

        

    

if __name__ == "__main__":
    main()
    
 
