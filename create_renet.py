from __future__ import print_function
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import caffe


def conv_relu(bottom, kernel_size, num_output, name, stride=1, pad=0, group=1):
    conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride,
                         num_output=num_output, pad=pad, group=group,
                             weight_filler=dict(type="xavier"),
                             bias_filler=dict(type="constant", value=0),
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             name="{0}_conv".format(name))
    relu = L.ReLU(conv, in_place=True,
                      name="{0}_relu".format(name))
    return relu


def conv_bn(bottom, kernel_size, num_output, name, deploy, stride=1, pad=0, group=1, conv_bias_term=True):
    if conv_bias_term:
        conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride,
                             num_output=num_output, pad=pad, group=group,
                             weight_filler=dict(type="xavier"),
                             bias_filler=dict(type="constant", value=0),
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             name="{0}_conv".format(name))
    else:
        conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride,
                             num_output=num_output, pad=pad, group=group,
                             weight_filler=dict(type="xavier"),
                             bias_term=False,
                             param=[dict(lr_mult=1, decay_mult=1)],
                             name="{0}_conv".format(name))


    if deploy:
        # In our BN layers, the provided mean and variance are strictly computed using
        # average (not moving average) on a sufficiently large training batch after the training procedure.
        # The numerical results are very stable (variation of val error < 0.1%).
        # Using moving average might lead to different results.
        # from https://github.com/KaimingHe/deep-residual-networks
        # So set use_global_stats = true in deployment. See also ReNet deployment.
        batch_norm = L.BatchNorm(conv, in_place=True,
                                     batch_norm_param=dict(use_global_stats=True),
                                     name="{0}_batch_norm".format(name))
    else:
        # By default, use_global_stats is set to false when the network is in the training
        # // phase and true when the network is in the testing phase.
        # from caffe BatchNorm
        batch_norm = L.BatchNorm(conv, in_place=True,
                                     param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0),
                                                         dict(lr_mult=0, decay_mult=0)],
                                 name="{0}_batch_norm".format(name))

    scale = L.Scale(batch_norm, bias_term=True, in_place=True,
                        name="{0}_scale".format(name))
    return scale

def conv_bn_relu(bottom, kernel_size, num_output, name, deploy, stride=1, pad=0, group=1, conv_bias_term=True):
    if conv_bias_term:
        conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride,
                             num_output=num_output, pad=pad, group=group,
                             weight_filler=dict(type="xavier"),
                             bias_filler=dict(type="constant", value=0),
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             name="{0}_conv".format(name))
    else:
        conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride,
                             num_output=num_output, pad=pad, group=group,
                             weight_filler=dict(type="xavier"),
                             bias_term=False,
                             param=[dict(lr_mult=1, decay_mult=1)],
                             name="{0}_conv".format(name))

    if deploy:
        # In our BN layers, the provided mean and variance are strictly computed using
        # average (not moving average) on a sufficiently large training batch after the training procedure.
        # The numerical results are very stable (variation of val error < 0.1%).
        # Using moving average might lead to different results.
        # from https://github.com/KaimingHe/deep-residual-networks
        # So set use_global_stats = true in deployment. See also ReNet deployment.
        batch_norm = L.BatchNorm(conv, in_place=True,
                                     batch_norm_param=dict(use_global_stats=True),
                                     name="{0}_batch_norm".format(name))
    else:
        # By default, use_global_stats is set to false when the network is in the training
        # // phase and true when the network is in the testing phase.
        # from caffe BatchNorm
        batch_norm = L.BatchNorm(conv, in_place=True,
                                     param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0),
                                                         dict(lr_mult=0, decay_mult=0)],
                                 name="{0}_batch_norm".format(name))

    scale = L.Scale(batch_norm, bias_term=True, in_place=True,
                        name="{0}_scale".format(name))
    relu = L.ReLU(scale, in_place=True,
                      name="{0}_relu".format(name))
    return relu

# without projection shortcut
# bottom shape equal to outputshape
def identity_residual_block(bottom, conv_nums,
                        deploy, name):
    conv3x3_1_num, conv3x3_2_num = conv_nums
    # 3x3
    conv3x3_1 = conv_bn_relu(bottom, kernel_size=1, stride=1, pad=0, conv_bias_term=False,
                                  num_output=conv3x3_1_num, deploy=deploy,
                                  name="{0}_conv3x3_1".format(name))

    # 3x3
    conv3x3_2 = conv_bn_relu(conv3x3_1, kernel_size=3, stride=1, pad=1, conv_bias_term=False,
                           num_output=conv3x3_2_num, deploy=deploy,
                           name="{0}_conv3x3_2".format(name))
    # elt_wise_add
    elt_wise_add = L.Eltwise(conv3x3_2, bottom,
                                 eltwise_param=dict(operation=P.Eltwise.SUM),
                                 name="{0}_elt_wise_add".format(name))

    # elt_wise_add_relu
    elt_wise_add_relu = L.ReLU(elt_wise_add, in_place=True,
                              name="{0}_elt_wise_add_relu".format(name))

    return elt_wise_add_relu

# with projection shortcut
def proj_residual_block(bottom, conv_nums, conv3x3_1_stride, conv1x1_proj_stride,
                        deploy, name):
    conv3x3_1_num, conv3x3_2_num = conv_nums

    # 3x3
    conv3x3_1 = conv_bn_relu(bottom, kernel_size=3, stride=conv3x3_1_stride, pad=1, conv_bias_term=False,
                                  num_output=conv3x3_1_num, deploy=deploy,
                                  name="{0}_conv3x3_1".format(name))

    # 3x3
    conv3x3_2 = conv_bn_relu(conv3x3_1, kernel_size=3, stride=1, pad=1, conv_bias_term=False,
                           num_output=conv3x3_2_num, deploy=deploy,
                                  name="{0}_conv3x3_2".format(name))

    # projection shutcut
    conv1x1_proj = conv_bn(bottom, kernel_size=1, stride=conv1x1_proj_stride, pad=0, conv_bias_term=False,
                           num_output=conv3x3_2_num, deploy=deploy,
                           name="{0}_conv1x1_proj".format(name))

    # elt_wise_add
    elt_wise_add = L.Eltwise(conv3x3_2, conv1x1_proj,
                                 eltwise_param=dict(operation=P.Eltwise.SUM),
                                 name="{0}_elt_wise_add".format(name))


    # elt_wise_add_relu
    elt_wise_add_relu = L.ReLU(elt_wise_add, in_place=True,
                              name="{0}_elt_wise_add_relu".format(name))

    return elt_wise_add_relu

# it can make cifar10 resnet20, resnet32, resnet44, resnet56
# n = {3; 5; 7; 9}
def make_resnet(input_shape, train_lmdb, test_lmdb, mean_file, conv2x_num,
           conv3x_num, conv4x_num, net_name,
           classes=10, deploy=False):
    if deploy:
        net_filename = "{0}_deploy.prototxt".format(net_name)
    else:
        net_filename = "{0}_train_test.prototxt".format(net_name)

    # net name
    with open(net_filename, "w") as f:
        f.write('name: "{0}"\n'.format(net_name))

    if deploy:
        """
        The conventional blob dimensions for batches of image data are 
        number N x channel K x height H x width W. Blob memory is row-major in layout, 
        so the last / rightmost dimension changes fastest. 
        For example, in a 4D blob, the value at index (n, k, h, w) is 
        physically located at index ((n * K + k) * H + h) * W + w.
        """
        # batch_size, channel, height, width
        data = L.Input(input_param=dict(shape=[dict(dim=list(input_shape))]), name="data")
    else:
        batch_size = 128
        data, label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=train_lmdb,
                             transform_param=dict(mirror=True,
                                                  crop_size=32,
                                                  mean_file=mean_file),
                                                  # mean_value=[104, 117, 123]),
                                                  ntop=2, include=dict(phase=caffe_pb2.Phase.Value("TRAIN")),
                                                  name="data")

        with open(net_filename, "a") as f:
            f.write(str(to_proto(data, label)))

        batch_size = 100
        data, label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=test_lmdb,
                                     transform_param=dict(mirror=False,
                                                          crop_size=32,
                                                          mean_file=mean_file),
                                                          # mean_value=[104, 117, 123]),
                                     ntop=2, include=dict(phase=caffe_pb2.Phase.Value("TEST")),
                             name="data")

    # conv1, output:32x32
    conv1 = conv_bn_relu(data, kernel_size=3, num_output=16, pad=1, stride=1, deploy=deploy, name="conv1")

    # conv2_x, output:32x32
    input = conv1
    conv_nums = [16, 16]
    for i in range(conv2x_num):
        output = identity_residual_block(input, conv_nums=conv_nums, deploy=deploy,
                                         name="conv2_{0}".format(i + 1))
        input = output


    # conv3_x, output:16x16
    conv_nums = [32, 32]
    for i in range(conv3x_num):
        if i == 0:
            output = proj_residual_block(input, conv_nums=conv_nums, conv3x3_1_stride=2,
                                         conv1x1_proj_stride=2, deploy=deploy,
                                         name="conv3_{0}".format(i + 1))
        else:
            output = identity_residual_block(input, conv_nums=conv_nums, deploy=deploy,
                                         name="conv3_{0}".format(i + 1))
        input = output

    # conv4_x, output:8x8
    conv_nums = [64, 64]
    for i in range(conv4x_num):
        if i == 0:
            output = proj_residual_block(input, conv_nums=conv_nums, conv3x3_1_stride=2,
                                         conv1x1_proj_stride=2, deploy=deploy,
                                         name="conv4_{0}".format(i + 1))
        else:
            output = identity_residual_block(input, conv_nums=conv_nums, deploy=deploy,
                                         name="conv4_{0}".format(i + 1))
        input = output

    # avg pool
    avgpool = L.Pooling(input, kernel_size=8, stride=1, pool=P.Pooling.AVE,
                              name="avgpool")

    # pred fc
    pred_fc = L.InnerProduct(avgpool, num_output=classes,
                                 weight_filler=dict(type="xavier"),
                                 bias_filler=dict(type="constant", value=0),
                                 param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             name="pred_fc")
    # loss
    if deploy:
        prob = L.Softmax(pred_fc, name="prob")
        with open(net_filename, "a") as f:
            f.write(str(to_proto(prob)))
    else:
        loss = L.SoftmaxWithLoss(pred_fc, label, name="loss")
        accuracy = L.Accuracy(pred_fc, label,
                                              include=dict(phase=caffe_pb2.Phase.Value('TEST')),
                                    name="accuracy")
        with open(net_filename, "a") as f:
            f.write(str(to_proto(loss, accuracy)))

# n = 3
def make_cifar10_resnet20(input_shape, train_lmdb, test_lmdb, mean_file, classes=10, deploy=False):
    make_resnet(input_shape, train_lmdb=train_lmdb, test_lmdb=test_lmdb, mean_file=mean_file,
                classes=classes, deploy=deploy, net_name="cifar10_resnet20",
                conv2x_num=3, conv3x_num=3, conv4x_num=3)

# n = 5
def make_cifar10_resnet32(input_shape, train_lmdb, test_lmdb, mean_file, classes=10, deploy=False):
    make_resnet(input_shape, train_lmdb=train_lmdb, test_lmdb=test_lmdb, mean_file=mean_file,
                classes=classes, deploy=deploy, net_name="cifar10_resnet32",
                conv2x_num=5, conv3x_num=5, conv4x_num=5)


# n = 7
def make_cifar10_resnet44(input_shape, train_lmdb, test_lmdb, mean_file, classes=10, deploy=False):
    make_resnet(input_shape, train_lmdb=train_lmdb, test_lmdb=test_lmdb, mean_file=mean_file,
                classes=classes, deploy=deploy, net_name="cifar10_resnet44",
                conv2x_num=7, conv3x_num=7, conv4x_num=7)

# n = 9
def make_cifar10_resnet56(input_shape, train_lmdb, test_lmdb, mean_file, classes=10, deploy=False):
    make_resnet(input_shape, train_lmdb=train_lmdb, test_lmdb=test_lmdb, mean_file=mean_file,
                classes=classes, deploy=deploy, net_name="cifar10_resnet56",
                conv2x_num=9, conv3x_num=9, conv4x_num=9)

# n = 18
def make_cifar10_resnet110(input_shape, train_lmdb, test_lmdb, mean_file, classes=10, deploy=False):
    make_resnet(input_shape, train_lmdb=train_lmdb, test_lmdb=test_lmdb, mean_file=mean_file,
                classes=classes, deploy=deploy, net_name="cifar10_resnet110",
                conv2x_num=18, conv3x_num=18, conv4x_num=18)

if __name__ == '__main__':
    input_shape = [1, 3, 32, 32]
    classes = 10

    train_lmdb = "/home/tim/datasets/cifar10/train_lmdb"
    test_lmdb = "/home/tim/datasets/cifar10/test_lmdb"
    mean_file = "/home/tim/datasets/cifar10/mean.binaryproto"

    make_cifar10_resnet20(input_shape=input_shape, train_lmdb=train_lmdb,
                  test_lmdb=test_lmdb, mean_file=mean_file, classes=classes, deploy=False)
    make_cifar10_resnet20(input_shape=input_shape, train_lmdb=None,
                  test_lmdb=None, mean_file=mean_file, classes=classes, deploy=True)

    net_name = "cifar10_resnet20"
    # caffe.Net("{0}_solver.prototxt".format(net_name))  # test loading the net

    solver = caffe.SGDSolver("{0}_solver.prototxt".format(net_name))

    for k, v in solver.net.blobs.items():
        print(k, v.data.shape)

    make_cifar10_resnet32(input_shape=input_shape, train_lmdb=train_lmdb,
                          test_lmdb=test_lmdb, mean_file=mean_file, classes=classes, deploy=False)
    make_cifar10_resnet32(input_shape=input_shape, train_lmdb=None,
                          test_lmdb=None, mean_file=mean_file, classes=classes, deploy=True)

    make_cifar10_resnet44(input_shape=input_shape, train_lmdb=train_lmdb,
                          test_lmdb=test_lmdb, mean_file=mean_file, classes=classes, deploy=False)
    make_cifar10_resnet44(input_shape=input_shape, train_lmdb=None,
                          test_lmdb=None, mean_file=mean_file, classes=classes, deploy=True)

    make_cifar10_resnet56(input_shape=input_shape, train_lmdb=train_lmdb,
                          test_lmdb=test_lmdb, mean_file=mean_file, classes=classes, deploy=False)
    make_cifar10_resnet56(input_shape=input_shape, train_lmdb=None,
                          test_lmdb=None, mean_file=mean_file, classes=classes, deploy=True)

    # RecursionError: maximum recursion depth exceeded
    import sys
    sys.setrecursionlimit(1000000)  # 例如这里设置为一百万

    make_cifar10_resnet110(input_shape=input_shape, train_lmdb=train_lmdb,
                          test_lmdb=test_lmdb, mean_file=mean_file, classes=classes, deploy=False)
    make_cifar10_resnet110(input_shape=input_shape, train_lmdb=None,
                          test_lmdb=None, mean_file=mean_file, classes=classes, deploy=True)