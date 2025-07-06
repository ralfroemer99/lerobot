from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

dataset = LeRobotDataset(repo_id="test_johannes", root="/home/ralf/.cache/huggingface/lerobot/test_johannes")

dataset.push_to_hub(repo_id="ralfroemer/pick_lego_block_test")