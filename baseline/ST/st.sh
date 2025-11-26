# datapath=/data/datasets/MVTecLoco/
datapath=/mnt/e/datasets/mvtec_loco/
datasets=('breakfast_box' 'juice_bottle' 'pushpins' 'screw_bag' 'splicing_connectors')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

python3 ./baseline/ST/main4st.py \
--gpu 1 \
--seed 42 \
--log_group st_mvtec_loco \
--log_project MVTecLOCO_Results \
--results_path results \
--run_name run \
stnet \
--patch_size 65 \
--n_students 3 \
dataset \
--batch_size 16 \
--resize 329 \
--imagesize 65 "${dataset_flags[@]}" mvtecloco $datapath
