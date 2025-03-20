python lerobot/scripts/train.py \
    --policy.path=lerobot/act_aloha_sim_transfer_cube_human \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --env.type=aloha \
    --env.task=AlohaInsertion-v0 \
    --steps=5000 \
    --eval_freq=1000 \
    --save_freq=1000