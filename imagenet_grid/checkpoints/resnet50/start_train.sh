python3 -W ignore -m torch.distributed.launch --nproc_per_node=8 imagenet_amp.py -a resnet50 \
    --opt-level O0 \
    --b 32 \
    -p 1000 \
    --epochs 300 \
    --save_dir checkpoints/resnet50_grid6 \
    -j 64 \
    --seed 5 \
    --lr_adjust_step 100 200 265 \
    --grid --d1 96 --d2 224 --ratio 0.6 --rotate 360 --mode 1 --prob 0.8 --st_epochs 240  \
    /data1/Data/ImageNet/

    #--resume checkpoints/resnet50_grid2/checkpoint.pth.tar \
    # no warm up
    # set 96-224 in grid.py
    
    # cuda V10.0.130
    # nvidia 418.67 P100
    # python 3.6.5 pytorch 1.3.0+cu100 torchvision 0.4.1+cu100
