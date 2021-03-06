# export PYTHONPATH=$(pwd)/../../second.pytorch/:$PYTHONPATH
export PYTHONPATH=$(pwd)/../:$PYTHONPATH

# python3 create_data.py kitti_data_prep --data_path=/kitti/

# python3 ./pytorch/train.py train --config_path=./configs/car.fhd.config --model_dir=second1 --resume=True

CUDA_VISIBLE_DEVICES=0,1 \
python3 ./pytorch/train.py train  \
<<<<<<< HEAD
 --config_path=configs/car.fhd.config --model_dir=depconv74 --resume=False \
 --multi_gpu=False
=======
 --config_path=configs/car.fhd.config --model_dir=depconv93 --resume=False\
 --multi_gpu=True
>>>>>>> depconv93

#CUDA_VISIBLE_DEVICES=2 python3 ./pytorch/train.py evaluate  \
 #--config_path=configs/car.fhd.config --model_dir=depconv26

# CUDA_VISIBLE_DEVICES=1 python3 ./pytorch/train.py train \
# --config_path=configs/pointpillars/car/xyres_16.config --model_dir=pp16 --resume=False \
# --multi_gpu=False

# python3 ./pytorch/train.py evaluate --config_path=./configs/car.fhd.config --model_dir=second1 --measure_time=True --batch_size=1
