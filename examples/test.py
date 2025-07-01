import numpy as np
import torch
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

policy = DiffusionPolicy.from_pretrained("/home/ralf/Projects/lerobot/outputs/train/dp_test_ralf/checkpoints/last/pretrained_model")
policy.reset()

state=np.zeros(7)
image=np.zeros((256, 256, 3))

observation = {
    "observation.state": state,
    "observation.images.right_wrist_camera": image,
}

# Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
for name in observation:
    observation[name] = torch.from_numpy(observation[name])
    if "image" in name:
        observation[name] = observation[name].type(torch.float32) / 255
        observation[name] = observation[name].permute(2, 0, 1).contiguous()
    else:
        observation[name] = observation[name].type(torch.float32)
    observation[name] = observation[name].unsqueeze(0)
    observation[name] = observation[name].to('cuda')

action = policy.select_action(observation)

