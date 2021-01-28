import unittest as ut, os, toml

from run_toml import ExperimentManager

class TestExperimentManager(ut.TestCase):
    def setUp(self):

        # dump fake settings file
        test_settings = dict(experiments = dict(test_experiment = dict()))
        with open("test_settings.toml") as f:
            toml.dump(test_settings, f)

        # load it
        self.manager = ExperimentManager("test_settings.toml")

    def test_run(self):
        self.manager.run()

    def tearDown(self):
        os.remove("test_settings.toml")
        

class TestDummyExperiment(ut.TestCase):
    def setUp(self):
        pass

class TestDummyWorker(ut.TestCase):
    def setUp(self):
        pass
