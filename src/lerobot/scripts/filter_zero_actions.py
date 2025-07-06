from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)

source_repo_id = "ralfroemer/pick_lego_block"
target_repo_id = source_repo_id + "_filtered"

# Copy all frames from the source dataset to the target dataset, but filter out frames with zero actions.
source_dataset = LeRobotDataset(repo_id=source_repo_id)


dataset = LeRobotDataset.create(
    repo_id=target_repo_id,
    fps=source_dataset.meta.fps,
    robot_type=source_dataset.meta.robot_type,
    features=source_dataset.meta.features,
    use_videos=False,
)

for episode in source_dataset.episodes:
    for frame in episode.frames:
        if frame["action"].sum() > 0:  # Only keep frames with non-zero actions
            dataset.add_frame(episode.id, frame)

dataset.push_to_hub(repo_id=target_repo_id, private=True)