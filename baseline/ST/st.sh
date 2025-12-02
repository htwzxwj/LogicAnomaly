datapath=/data/datasets/MVTecLoco/
# datapath=/mnt/e/datasets/mvtec_loco/ # datapath for WSL
datasets=('breakfast_box' 'juice_bottle' 'pushpins' 'screw_bag' 'splicing_connectors')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

python3 ./baseline/ST/main4st.py \
--gpu 1 \
--seed 42 \
--log_group st_mvtec_loco \
--log_project MVTecLOCO_Results \
--results_path results \
--run_name run \
--teacher_epochs 1 \
--student_epochs 1 \
stnet \
--patch_size 65 \
--n_students 3 \
dataset \
--batch_size 3 \
--imagesize 256 "${dataset_flags[@]}" mvtecloco $datapath
