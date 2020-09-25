from plexsim.models import *
import toml, pickl
import sys; sys.path.insert(0, '../')
from Utils.graph import *
from Toolbox import infcy

class toml_reader:
    def __init__(self, fn):
        self.settings = toml.load(fn)
        self.sim_settings = self.settings.get('simulation')
    
    def setup_model(self):
        model_settings = self.settings.get('model')
        name = model_settings.get('name', '')
        settings = model_settings.get('settings', {})
        
        g_name = model_settings.get('graph').get('name')
        g_settings = model_settings.get('graph').get('settings', {})
        g = globals()[g_name](**g_settings)
        m = globals()[name](g, **settings)
        return m

reader = toml_reader('run_settings.toml')
m = reader.setup_model()
SIM = infcy.Simulator(m)
snapshots = SIM.snapshots(**reader.sim_settings.get('snapshots'))
output = SIM.forward(snapshots, **reader.sim_settings.get('conditional'))
px, mi = infcy.mutualInformation(output.get('conditional'), output.get('snapshots'))

fn = 'testing.pickle'
with open('data/{fn}') as f:
     o = dict(model = m, output = output, mi = mi, px = px)
     pickle.dump(f, o)

print('exited')
