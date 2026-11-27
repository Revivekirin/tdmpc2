"""Historical task and diagnostic contracts; no experiment runs on import."""
from pathlib import Path
import sys
SCRIPTS=Path(__file__).resolve().parents[2]/"scripts"
sys.path.insert(0,str(SCRIPTS))
GRID = (.5, .75, 1., 1.25, 1.5, 2.)
METHODS = ('F', 'D', 'DQ', 'J')
LABEL = {'F':'Frozen', 'D':'D', 'DQ':'D+Q', 'J':'Joint', 'DQ_F':'D+Q predictor / Frozen Q'}
COLORS = {'F':'#333333', 'D':'#0072B2', 'DQ':'#D55E00', 'J':'#009E73', 'DQ_F':'#CC79A7'}
PAIRS = [('F','D'), ('D','DQ'), ('F','J'), ('DQ','J'), ('DQ_F','DQ')]
TASKS = {
 'halfcheetah':dict(families={'damping':'bthigh_damping','mass':'torso_mass'},
  adaptation=list(range(42000001,42000004)), evaluation=list(range(43000001,43000021)),
  checkpoint='nominal-server4-20261108-nohup-01/checkpoint-000750000', budget=3000,
  sha='f44afd1992e194fd0d70b766398de7ad3d0ca880c55b46e3b8c34ad9dade013b'),
 'reacher':dict(families={'D0':'joint0_damping','D1':'joint1_damping','M0':'body0_mass','M1':'body1_mass'},
  adaptation=list(range(53000001,53000004)), evaluation=list(range(54000001,54000021)),
  checkpoint='reacher-full-01/checkpoint-000670000', budget=10000,
  sha='218624bfb833bfa7f732c66ad94b8e8b80b30efb34e5ac67776b58347b1e333e')}

