conda activate downscaling

CUDA_VISIBLE_DEVICES=0 python3 train.py --root_file data/WindTopoLarge/ --target_file sample.csv --epoch 100 --batch_size 512 --device cuda --exp_name small_exp_lr_100epochs