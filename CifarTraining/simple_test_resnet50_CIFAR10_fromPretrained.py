from typing import Type, Union, List, Optional
import mindspore.nn as nn
from mindspore.common.initializer import Normal

import os

import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore import dtype as mstype

from mindspore import train
from mindspore.train import Model, Callback, LossMonitor
from resnet import resnet50

ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU")

def create_dataset_cifar10(dataset_dir, usage, resize, batch_size, workers):

    data_set = ds.Cifar10Dataset(dataset_dir=dataset_dir,
                                 usage=usage,
                                 num_parallel_workers=workers,
                                 shuffle=True)

    trans = []
    if usage == "train":
        trans += [
            vision.RandomCrop((32, 32), (4, 4, 4, 4)),
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
    data_set = data_set.batch(batch_size)

    return data_set



data_dir = "/local1/ctribes/mindspore/datasets-cifar10-bin/cifar-10-batches-bin"  # Root directory of the dataset
batch_size = 256  # Batch size
image_size = 32  # Image size of training data
workers = 4  # Number of parallel workers
num_classes = 10  # Number of classes
num_epochs = 100


# Obtain the preprocessed training and testing datasets

dataset_train = create_dataset_cifar10(dataset_dir=data_dir,
                                       usage="train",
                                       resize=image_size,
                                       batch_size=batch_size,
                                       workers=workers)
step_size_train = dataset_train.get_dataset_size()

dataset_val = create_dataset_cifar10(dataset_dir=data_dir,
                                     usage="test",
                                     resize=image_size,
                                     batch_size=batch_size,
                                     workers=workers)
step_size_val = dataset_val.get_dataset_size()



resnet50_ckpt = "./resnet50.ckpt"
resnet50_new_ckpt = "./resnet50_new.ckpt"
pretrained = True
network = resnet50(pretrained, resnet50_ckpt)


# Size of the input layer of the fully-connected layer
in_channel = network.fc.in_channels
fc = nn.Dense(in_channels=in_channel, out_channels=num_classes)
# Reset the fully-connected layer.
network.fc = fc


# Set the learning rate
lr = nn.cosine_decay_lr(min_lr=0.00001, max_lr=0.001, total_step=step_size_train * num_epochs,
                        step_per_epoch=step_size_train, decay_epoch=num_epochs)
# Define optimizer and loss function
opt = nn.Momentum(params=network.trainable_params(), learning_rate=lr, momentum=0.9)
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')


# Instantiate models
model = ms.Model(network, loss_fn=loss_fn, optimizer=opt, metrics={'acc'})
print('Model is done')

from mindspore.train.callback._callback import Callback, _handle_loss
class PrintInfo(Callback):
    def on_train_epoch_end(self,run_context):
        cb_params = run_context.original_args()
        #metrics = cb_params.get("metrics")
        loss = _handle_loss(cb_params.net_outputs)
        #out = model.eval(dataset_val)
        print("Eval result: epoch %d, loss: %s" % (cb_params.cur_epoch_num, loss))


print_cb=PrintInfo()
print("Start Training Loop ...")
model.train(num_epochs, dataset_train, callbacks=print_cb)

# Save checkpoint
print("Save checkpoint")
ms.save_checkpoint(network,resnet50_new_ckpt)


out = model.eval(dataset_val)
print("Eval out: ", out)

