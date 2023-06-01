import argparse
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
import mindspore as ms
from mindspore.communication import init, get_rank, get_group_size
import numpy as np
# import download
import os
from mindspore import dtype as mstype
from mindspore import nn, ops
from mindspore import load_checkpoint, load_param_into_net
from mindspore import train
from mindspore.train import Model, Callback, LossMonitor
from resnet import resnet50

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--run_distribute', action='store_true', help='Run distribute.')
parser.add_argument('--device_num', type=int, default=1, help='Device num.')
parser.add_argument('--do_train', action='store_true', help='Do train or not.')
parser.add_argument('--do_eval', action='store_true', help='Do eval or not.')
parser.add_argument('--epoch_size', type=int, default=1, help='Epoch size.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
parser.add_argument('--num_classes', type=int, default=10, help='Num classes.')
parser.add_argument('--checkpoint_path', type=str, default=None, help='CheckPoint file path.')
parser.add_argument('--dataset_path', type=str, default=None, required=True, help='Dataset path.')
args_opt = parser.parse_args()

data_home = args_opt.dataset_path

ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU")
init("nccl")
#ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.AUTO_PARALLEL, gradients_mean=True)
ms.set_auto_parallel_context(device_num=args_opt.device_num, parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                              parameter_broadcast=True, all_reduce_fusion_config=[140])

def create_dataset_cifar10(usage, resize, workers):

    assert os.path.exists(data_home), "the dataset path is invalid!"

    if args_opt.run_distribute:
        rank_id = int(os.getenv('RANK_ID'))
        rank_size = int(os.getenv('RANK_SIZE'))
        data_set = ds.Cifar10Dataset(dataset_dir=data_home, 
                                    usage=usage, 
                                    num_shards=rank_size, 
                                    shard_id=rank_id,
                                    num_parallel_workers=workers,
                                    shuffle=True)
    else:
        data_set = ds.Cifar10Dataset(dataset_dir=data_home,
                                    usage=usage,
                                    num_parallel_workers=workers,
                                    shuffle=True)

    trans = []
    if usage == "train":
        trans += [
            vision.RandomCrop((32, 32), (4, 4, 4, 4),pad_if_needed=True), # padding_mode default CONSTANT
            vision.RandomHorizontalFlip(prob=0.5)
        ]

    trans += [
        vision.Resize(resize),
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        vision.HWC2CHW()
    ]

    target_trans = transforms.TypeCast(mstype.int32)

    # Data transformation
    data_set = data_set.map(operations=trans,
                            input_columns='image',
                            num_parallel_workers=workers)

    data_set = data_set.map(operations=target_trans,
                            input_columns='label',
                            num_parallel_workers=workers)

    # Batching
    data_set = data_set.batch(batch_size=args_opt.batch_size, drop_remainder=True)

    return data_set



resnet50_ckpt = args_opt.checkpoint_path
pretrained = resnet50_ckpt is not None
network = resnet50(pretrained, resnet50_ckpt)


##### Reset if num_classes change 
# Size of the input layer of the fully-connected layer
#in_channel = network.fc.in_channels
#fc = nn.Dense(in_channels=in_channel, out_channels=args_opt.num_classes)
# Reset the fully-connected layer.
#network.fc = fc

for param in network.get_parameters():
    param.requires_grad = True


# Obtain the processed training and test datasets
image_size = 32  # Image size of training data
workers = 4  # Number of parallel workers
if args_opt.do_train:
    dataset_train = create_dataset_cifar10( usage="train",
                                            resize=image_size,
                                            workers=workers)
    step_size_train = dataset_train.get_dataset_size()

if args_opt.do_eval:
    dataset_val = create_dataset_cifar10( usage="test",
                                        resize=image_size,
                                        workers=workers)
    step_size_train=1 # this is used by lr cosine decay but if do_eval only, it is not used


# Set the learning rate
epoch_size = args_opt.epoch_size
lr = nn.cosine_decay_lr(min_lr=0.00001, max_lr=0.001, total_step=step_size_train * epoch_size,
                        step_per_epoch=step_size_train, decay_epoch=epoch_size)
# Define optimizer and loss function
opt = nn.Momentum(params=network.trainable_params(), learning_rate=lr, momentum=0.9)
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')


# Instantiate models
model = ms.Model(network, loss_fn=loss_fn, optimizer=opt, metrics={'acc'})
print('Model is done')

# as for train, users could use model.train
if args_opt.do_train:
    loss_cb = train.LossMonitor()
    print("Start Training Loop ...")
    model.train(epoch_size, dataset_train, callbacks=loss_cb)
    if args_opt.checkpoint_path is not None:
        print("Save checkpoint")
        ms.save_checkpoint(network,"new_checkpoint.ckpt")


# as for evaluation, users could use model.eval
if args_opt.do_eval:
    out = model.eval(dataset_val)
    print("Eval out: ", out)


