********************************************************************
# Training ResNet50 on ImageNet(2012) dataset
********************************************************************

What is needed: 
- Image dataset. 
- Mindspore v2.0
- GPU

How to run in parallel (MPI):
CUDA_VISIBLE_DEVICES=1,2,3,4 nohup bash -c 'mpirun -np 4 python3 simple_test_imagenet_gpu_parallel.py --dataset_path=/local1/datasets/imagenet/ILSVRC2012 --epoch_size=100 --batch_size=128 --num_classes=1000 --device_num=4 --checkpoint_path=./resnet50.ckpt'&

Checkpoint path is optional.
