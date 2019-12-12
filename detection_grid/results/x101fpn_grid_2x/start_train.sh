NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_net.py \
	--config-file configs/grid_faster_rcnn_X_101_32x8d_FPN_2x.yaml
