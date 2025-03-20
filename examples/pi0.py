from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy

# import torch
# torch.cuda.empty_cache()
# torch.cuda.reset_peak_memory_stats()


policy = PI0Policy.from_pretrained("lerobot/pi0")

policy.save_pretrained("/home/ralf_roemer/Projects/models/pi0")