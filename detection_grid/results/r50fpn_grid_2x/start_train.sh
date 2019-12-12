NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_net.py \
	--config-file configs/grid_faster_rcnn_R_50_FPN_2x.yaml
