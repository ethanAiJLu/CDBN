python train_sfda.py --trainer Source_adapter --dataset-config-file /home/CDBN/configs/datasets/office_homea2p.yaml --config-file /home/CDBN/configs/trainers/rn50.yaml --output-dir /home/CDBN/output/office/source --cls-rate 0.2 --num-shots-source 8

python generate_label.py --eval-only --trainer Source_adapter --dataset-config-file /home/CDBN/configs/datasets/office_homea2p.yaml --init-weights /home/CDBN/output/office/source

python train_sfda.py --trainer Clip_adapter_target --dataset-config-file /home/CDBN/configs/datasets/office_homea2p.yaml --config-file /home/CDBN/configs/trainers/rn50.yaml --output-dir /home/CDBN/output/office/rn50 --init-weights /home/CDBN/output/office/source --cls-rate 0.2 --fixmatch 1.0 --im-weight 0.0 --im-minus 1.0 --parameter-alpha 0.5 --xshots 8 --freeze-class 
