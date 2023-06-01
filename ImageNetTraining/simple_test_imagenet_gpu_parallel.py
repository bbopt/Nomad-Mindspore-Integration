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
from mindspore.train import Model, Callback, LossMonitor
from mindspore import load_checkpoint, load_param_into_net
from resnet import resnet50
#from models import *


parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--device_num', type=int, default=1, help='Device num.')
parser.add_argument('--epoch_size', type=int, default=1, help='Epoch size.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
parser.add_argument('--num_classes', type=int, default=10, help='Num classes.')
parser.add_argument('--checkpoint_path', type=str, default=None, help='CheckPoint file path.')
parser.add_argument('--dataset_path', type=str, default=None, required=True, help='Dataset path.')
args_opt = parser.parse_args()


ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU")

# Parallel mode only
init("nccl")
ms.set_auto_parallel_context(device_num=args_opt.device_num, parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                              parameter_broadcast=True, all_reduce_fusion_config=[140])


def create_dataset(repeat_num=1, training=True):
    """
    create data for next use such as training or inferring
    """

    if training:
        data_home = args_opt.dataset_path+'/train'
    else:
        data_home = args_opt.dataset_path+'/val'


    assert os.path.exists(data_home), "the dataset path is invalid!"


    rank_id = int(os.getenv('RANK_ID'))
    rank_size = int(os.getenv('RANK_SIZE'))
    imagenet_ds = ds.ImageFolderDataset(data_home, num_shards=rank_size, shard_id=rank_id, decode=True)

    resize_height = 224
    resize_width = 224
    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    random_crop_op = vision.RandomCrop((32, 32), (4, 4, 4, 4),pad_if_needed=True) # padding_mode default CONSTANT
    random_horizontal_op = vision.RandomHorizontalFlip()
    resize_op = vision.Resize((resize_height, resize_width)) # interpolation default BILINEAR
    rescale_op = vision.Rescale(rescale, shift)
    normalize_op = vision.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    changeswap_op = vision.HWC2CHW()
    type_cast_op = transforms.TypeCast(ms.int32)

    c_trans = []
    if training:
        c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op,
                changeswap_op]

    # apply map operations on images
    imagenet_ds = imagenet_ds.map(operations=type_cast_op, input_columns="label")
    imagenet_ds = imagenet_ds.map(operations=c_trans, input_columns="image")

    # apply shuffle operations
    imagenet_ds = imagenet_ds.shuffle(buffer_size=10)

    # apply batch operations
    imagenet_ds = imagenet_ds.batch(batch_size=args_opt.batch_size, drop_remainder=True)

    # apply repeat operations
    imagenet_ds = imagenet_ds.repeat(repeat_num)


    return imagenet_ds

# Obtain the processed training and test datasets

dataset_train = create_dataset(training=True)

step_size_train = dataset_train.get_dataset_size()
index_label_dict = dataset_train.get_class_indexing()

dataset_val = create_dataset(training=False)
step_size_val = dataset_val.get_dataset_size()


# The pretrained ResNet was for 10 classes. Instantiate with 10 classes first.
#network = resnet50(args_opt.batch_size, 10)
# The pretrained ResNet was for 1000 classes. 
network = resnet50(args_opt.batch_size, args_opt.num_classes)

if args_opt.checkpoint_path:
    assert os.path.exists(args_opt.checkpoint_path), "the checkpoint path is invalid!"
    # load pre-trained models
    param_dict = load_checkpoint(args_opt.checkpoint_path)
    # print(param_dict)
    load_param_into_net(network, param_dict)
    print("Pretrained model loaded")

# If the pretrained ResNet was for 10 classes. Need to reset.
#in_channel = network.fc.in_channels
#fc = nn.Dense(in_channels=in_channel, out_channels=args_opt.num_classes)
# Reset the fully-connected layer.
#network.fc = fc

for param in network.get_parameters():
    param.requires_grad = True


# Set the learning rate
epoch_size = args_opt.epoch_size
lr = nn.cosine_decay_lr(min_lr=0.000001, max_lr=0.5, total_step=step_size_train * epoch_size,
                        step_per_epoch=step_size_train, decay_epoch=epoch_size)
# Define optimizer and loss function
# opt = nn.Momentum(params=network.trainable_params(), learning_rate=lr, momentum=0.9)
opt = nn.Adagrad(params=network.trainable_params(), learning_rate=lr)
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')


# Instantiate models
model = ms.Model(network, loss_fn=loss_fn, optimizer=opt, metrics={'acc'})
print('Model for training is done')


# Start circuit training
print("Start Training Loop ...")


from mindspore.train.callback._callback import Callback, _handle_loss
class PrintInfo2(Callback):
    def on_train_epoch_end(self,run_context):
        cb_params = run_context.original_args()
        loss = _handle_loss(cb_params.net_outputs)
        print("Eval result: epoch %d, loss: %s" % (cb_params.cur_epoch_num, loss))


print_cb=PrintInfo2()
loss = model.train(epoch_size,dataset_train, callbacks=print_cb)

out = model.eval(dataset_val)
print("Eval out: ",out)

ms.save_checkpoint(network, "resnet50_new.ckpt")
