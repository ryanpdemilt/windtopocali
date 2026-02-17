conda activate downscaling

CUDA_VISIBLE_DEVICES=0 python3 train.py --root_file data/WindTopo40k/ --target_file sample.csv --epoch 100 --batch_size 128 --device cuda --exp_name big_exp_lr_100epochs --loss_name mse