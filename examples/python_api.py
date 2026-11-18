"""Minimal library use after running the demo from the repository root."""
import json
import torch

from physcast_yaw import load_config
from physcast_yaw.control import ControllerSettings, optimize_yaw
from physcast_yaw.learning import WakePredictor

torch.set_num_threads(4)
farm, wind, controller = load_config('configs/three_turbines.json')
model = WakePredictor('outputs/demo/model/model.pt')
result = optimize_yaw(farm, wind, model, current_heading_deg=[270.]*farm.n,
                      settings=ControllerSettings(**controller))
print(json.dumps(result, indent=2))
