python lerobot/scripts/train.py --dataset.repo_id=lerobot/pusht \
                                --policy.type=diffusion \
                                --env.type=pusht \
                                --steps=5000 \
                                --eval_freq=1000 \
                                --save_freq=1000