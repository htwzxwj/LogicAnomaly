datapath=/data/datasets/MVTecLoco/
datasets=('breakfast_box' 'juice_bottle' 'pushpins' 'screw_bag' 'splicing_connectors')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

python3 main.py \
--gpu 1 \
--seed 42 \
--log_group simplenet_mvtec_loco \
--log_project MVTecLOCO_Results \
--results_path results \
--run_name run \
net \
-b wideresnet50 \
-le layer2 \
-le layer3 \
--pretrain_embed_dimension 1536 \
--target_embed_dimension 1536 \
--patchsize 3 \
--meta_epochs 40 \
--embedding_size 256 \
--gan_epochs 3 \
--noise_std 0.015 \
--dsc_hidden 1024 \
--dsc_layers 2 \
--dsc_margin .5 \
--pre_proj 1 \
dataset \
--batch_size 16 \
--resize 329 \
--imagesize 288 "${dataset_flags[@]}" mvtecloco $datapath
