import sys
import os.path
import torch

code_path = os.path.dirname(os.path.abspath(__file__)) + "/"
sys.path.append(code_path)
msst_webui_path = os.path.abspath(os.path.join(code_path, "..", ".."))
if msst_webui_path not in sys.path:
	sys.path.append(msst_webui_path)

import yaml
from utils.attrdict import AttrDict

torch.set_float32_matmul_precision("medium")


def get_model(config_path, weights_path, device):
	from modules.bandit.core.model import MultiMaskMultiSourceBandSplitRNNSimple

	f = open(config_path)
	config = AttrDict(yaml.load(f, Loader=yaml.FullLoader))
	f.close()

	model = MultiMaskMultiSourceBandSplitRNNSimple(**config.model)
	d = torch.load(code_path + "model_bandit_plus_dnr_sdr_11.47.chpt")
	model.load_state_dict(d)
	model.to(device)
	return model, config
