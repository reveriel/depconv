export PYTHONPATH=/code/:$PYTHONPATH
# python3 create_data.py kitti_data_prep --data_path=/kitti/

#  CUDA_VISIBLE_DEVICES=3 python3 ./pytorch/train.py train --config_path=./configs/car.fhd.config --model_dir=second1 --resume=True

CUDA_VISIBLE_DEVICES=2 python3 ./pytorch/train.py train --config_path=./configs/car.fhd.config --model_dir=depconv --resume=True

# CUDA_VISIBLE_DEVICES=3 python3 ./pytorch/train.py evaluate --config_path=./configs/car.fhd.config --model_dir=second1 --measure_time=True --batch_size=1
