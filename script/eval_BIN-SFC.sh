#!/usr/bin/env bash
gpus=0

data_name=LEVIR #LEVIR, DSIFN ,CLCD-256,EGY_BCD
net_G=BIN_SFC 
split=test
checkpoint_root=output/checkpoints
vis_root=output/vis

project_name=BIN_SFC_LEVIR_b4_optimizer_sgd_lossce_lr0.01_LW0.01
checkpoint_name=best_ckpt.pt
img_size=256


python eval.py --split ${split} --net_G ${net_G} --img_size ${img_size} --vis_root ${vis_root} --checkpoints_root ${checkpoint_root} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}
