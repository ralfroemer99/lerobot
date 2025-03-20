# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python lerobot/scripts/eval.py \
    --policy.path=/home/ralf_roemer/Projects/models/pi0 \
    --env.type=aloha \
    --env.task=AlohaInsertion-v0 \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
