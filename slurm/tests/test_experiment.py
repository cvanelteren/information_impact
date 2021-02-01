import os,time, sys; sys.path.insert(0, '../')
from unittest import TestCase, main
from Task import Task

SETTINGS = "tests/test_settings.toml"

from run import ExperimentManager

class TestExperimentManager(TestCase):
    """
    A manager should:
    - manage workers
    - dump to disk
    - restart
    """
    def setUp(self):
        # print("Testing setup")
        self.manager = ExperimentManager(SETTINGS, timeout = 5)

    def test_destination_directory(self):
        self.assertTrue(os.path.isdir(self.manager.base))

    # check if it runs
    def test_run(self):
        # print("Testing run")
        self.assertEqual(self.manager.run(), None) 

    # def test_dump_to_disk(self):
    #     print("Testing dumping to disk")
    #     self.manager.dump_to_disk()
    #     self.assertTrue(os.path.exists(self.manager.name + ".pickle"))
    # # check restart
    # def test_restart(self):
    #     print("Testing restart")
    #     self.manager.restart()
    #     self.assertFalse(self.manager.is_running)
    #     fp = self.manager.name + ".pickle"

    #     self.assertTrue(os.path.exists(fp))
        

    # def test_no_created_files(self):
    #    fp = self.manager.name + ".pickle"
    #    if os.path.exists(fp):
    #        os.remove(fp)

    # check if no files are left behind
    # def tearDown(self):
       # self.test_no_created_files()

# from slurm.tests import sample_experiment
# from . import sample_experiment
# class TestExperiment(TestCase):
#     """
#     An experiment should
#     - setup
#     - implement a task
#     - run
#     """
#     def test_setup(self):
#         # check if zero input raises
#         with self.assertRaises(TypeError):
#             sample_experiment.setup()
#         # check the output
#         tasks = sample_experiment.setup({})

#         self.assertEqual(type(tasks), list)
#         # testing magic number (see experiment file)
#         self.assertEqual(len(tasks), 1)

#         # check if all inputs are Task
#         checks = (isinstance(t, sample_experiment.Experiment) for t in tasks) 
#         self.assertTrue(all(checks))
#         self.tasks = tasks

#     def test_run(self):
#         results = []
#         tasks = sample_experiment.setup({})
#         for t in tasks:
#             r = t.run()
#             self.assertEqual(r, {"hello there" : "how do you do?"})
        

# from Task import Worker
# from . import sample_experiment 
# class TestWorker(TestCase):
#     """
#     A worker needs to:
#     - hold tasks
#     - save results
#     - restart
#     """
#     def setUp(self):

#         experiment_settings = dict(test = {})
#         worker_settings = dict(
#                                 deadline = time.time() + 100,
#                                id = "!test",
#                                )
#         self.config = dict(
#             experiment_settings = experiment_settings,
#             worker_settings = worker_settings
#         )

#     def test_run(self):
#         worker = Worker(sample_experiment, self.config)
#         worker.setup_experiment()
#         tasks = worker.tasks.copy()
#         # should finish
#         self.assertEqual(worker.run(), None)
#         for t in tasks:
#             fp = worker.create_output_file(t)
#             fp = os.path.join(t.output_directory, fp)
#             self.assertTrue(os.path.exists(fp))
        
#     def test_restart(self):
#         worker = Worker(sample_experiment, self.config)
#         worker.restart()
#         self.assertEqual(worker.tasks_done, False)

#         worker.clean_up()
#         fp = worker.create_dump_name()
#         self.assertFalse(os.path.exists(fp))
    

if __name__ == "__main__":
    main()
        
