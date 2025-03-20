export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python lerobot/scripts/train.py \
    --policy.path=/home/ralf_roemer/Projects/models/pi0 \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --env.type=aloha \
    --env.task=AlohaInsertion-v0 \
    --batch_size=4 \
    --steps=5000 \
    --eval_freq=1000 \
    --save_freq=1000