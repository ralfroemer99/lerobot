import numpy as np
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

# policy = DiffusionPolicy.from_pretrained("/home/ralf/Projects/lerobot/outputs/train/dp_test_ralf/checkpoints/last/pretrained_model")
policy = SmolVLAPolicy.from_pretrained("/home/ralf/Projects/lerobot/outputs/train/smolvla_test_ralf/checkpoints/last/pretrained_model")
policy.reset()
state=np.zeros(7)
image=np.zeros((256, 256, 3))

observation = {
    "observation.state": state,
    "observation.images.right_wrist_camera": image,
    "task": "pick the lego block."
}

# Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
for name in observation:
    if isinstance(observation[name], np.ndarray):
        observation[name] = torch.from_numpy(observation[name])
    
    if "image" in name:
        observation[name] = observation[name].type(torch.float32) / 255
        observation[name] = observation[name].permute(2, 0, 1).contiguous()
    elif "state" in name:
        observation[name] = observation[name].type(torch.float32)
    
    if not "task" in name:
        observation[name] = observation[name].unsqueeze(0)
        observation[name] = observation[name].to('cuda')

action = policy.select_action(observation)

print("Action:", action)

