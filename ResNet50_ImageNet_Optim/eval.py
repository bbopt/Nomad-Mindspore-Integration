import argparse
import numpy as np
import os

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
import mindspore as ms
from mindspore.communication import init, get_rank, get_group_size

from mindspore import dtype as mstype
from mindspore import nn, ops
from mindspore.train import Model, Callback, LossMonitor
from mindspore import load_checkpoint, load_param_into_net


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
parser.add_argument('--min_lr', type=float, default=0.00001, help='Min lr for cosine decay.')
parser.add_argument('--max_lr', type=float, default=0.001, help='Max lr for cosine decay.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer.')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Max lr for cosine decay.')
parser.add_argument('--optimizer', type=str, default="Momentum", help='Optimizer.')

args_opt = parser.parse_args()
data_home = args_opt.dataset_path

ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU")


if args_opt.run_distribute:
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

    if args_opt.run_distribute:
        rank_id = int(os.getenv('RANK_ID'))
        rank_size = int(os.getenv('RANK_SIZE'))
        imagenet_ds = ds.ImageFolderDataset(data_home, num_shards=rank_size, shard_id=rank_id, decode=True)
    else:
        print('INF\n')
        exit()

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

network = resnet50(args_opt.batch_size, args_opt.num_classes)

#if os.path.exists("./resnet50.ckpt"):
#    # load pre-trained models
#    param_dict = load_checkpoint("./resnet50.ckpt")
#    # print(param_dict)
#    load_param_into_net(network, param_dict)
#    print("Pretrained model loaded")
#else:
print("No pretrained model loaded")


# Size of the input layer of the fully-connected layer
#in_channel = network.fc.in_channels
#fc = nn.Dense(in_channels=in_channel, out_channels=args_opt.num_classes)
# Reset the fully-connected layer.
#network.fc = fc

for param in network.get_parameters():
    param.requires_grad = True


# Set the learning rate
epoch_size = args_opt.epoch_size
min_lr = args_opt.min_lr
max_lr = args_opt.max_lr
assert min_lr < max_lr, "min lr and max lr are incompatible!"

lr = nn.cosine_decay_lr(min_lr=min_lr, max_lr=max_lr, total_step=step_size_train * epoch_size,
                        step_per_epoch=step_size_train, decay_epoch=epoch_size)


# Define optimizer and loss function
momentum = args_opt.momentum
weight_decay = args_opt.weight_decay

if args_opt.optimizer=="Momentum":
    opt = nn.Momentum(params=network.trainable_params(), learning_rate=lr, momentum=momentum, weight_decay=weight_decay)
elif args_opt.optimizer=="SGD":    
    opt = nn.SGD(params=network.trainable_params(), learning_rate=lr, momentum=momentum, weight_decay=weight_decay)
elif args_opt.optimizer=="Adagrad":    
    opt = nn.Adagrad(params=network.trainable_params(), learning_rate=lr, weight_decay=weight_decay)
elif args_opt.optimizer=="Adam":    
    opt = nn.Adam(params=network.trainable_params(), learning_rate=lr, momentum=momentum, weight_decay=weight_decay)
elif args_opt.optimizer=="Adamax":    
    opt = nn.AdaMax(params=network.trainable_params(), learning_rate=lr, weight_decay=weight_decay)
else:
    print('Optimizer not available: ',args_opt.optimizer)
    exit()

loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')


# Instantiate models
model = ms.Model(network, loss_fn=loss_fn, optimizer=opt, metrics={'acc'})
print('Model for training is done')


# Start circuit training
print("Start Training Loop ...")


class Print_info(Callback):
    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        print("Epoch_num: ", cb_params.cur_epoch_num)

from mindspore.train.callback._callback import Callback, _handle_loss
class PrintInfo2(Callback):
    def on_train_epoch_end(self,run_context):
        cb_params = run_context.original_args()
        #metrics = cb_params.get("metrics")
        loss = _handle_loss(cb_params.net_outputs)
        #out = model.eval(dataset_val)
        print("Eval result: epoch %d, loss: %s" % (cb_params.cur_epoch_num, loss))

#print_cb=Print_info()
#print_cb = LossMonitor()
print_cb=PrintInfo2()
loss = model.train(epoch_size,dataset_train, callbacks=print_cb)

# Save checkpoint
# ms.save_checkpoint(network,"resnet50.ckpt")


out = model.eval(dataset_val)
print("Eval_out=",-out['acc'])


