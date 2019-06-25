import unittest, sys, os;
if 'test' not in os.curdir:
    sys.path.insert(0, '../')
from Toolbox import infcy as information
import numpy as np
class TestInformation(unittest.TestCase):
    def test_entropy(self):
        """
        Test that fair coin gives 1 bit information
        """
        p = np.asarray([.5, .5])
        H = information.entropy(p)
        self.assertEqual(H, 1)

    def test_mutual_information(self):
        """
        Check whether random data gives 0 mutual information
        """
        deltas, nodes, states = (1, 1, 2)
        dist = np.random.choice([.5], size = (deltas, nodes, states))
        conditional = {\
                '0' : dist, '1' : dist,\
                }
        snapshots  = {i : 1 / len(conditional) for i in conditional}
        px, mi = information.mutualInformation(conditional, snapshots)
        self.assertEqual(mi, 0)
