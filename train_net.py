import caffe
import os
import cv2
import sys

from pylab import *

snapshot_dir = "snapshot/"
if not os.path.exists(snapshot_dir):
    os.mkdir(snapshot_dir)

def train_py(net_name):
    # caffe.set_device(0)
    # caffe.set_mode_gpu()

    solver = caffe.SGDSolver("{0}_solver.prototxt".format(net_name))

    for k, v in solver.net.blobs.items():
        print(k, v.data.shape)

    solver.solve()

def train_cmd(net_name):
    print("start train...")
    cmd = "caffe train --solver={0}_solver.prototxt 2>&1|tee {0}_train.log".format(net_name)
    print(cmd)
    os.system(cmd)

def get_latest_snapshot(net_name):
    # lenet_iter_2788.solverstate
    filelist = os.listdir(snapshot_dir)
    latest_snapshot = None
    max_num = 0
    for file in filelist:
        if net_name in file:
            if ".solverstate" in file:
                num = int(file.split(".solverstate")[0].split("iter_")[-1])
                if num > max_num:
                    max_num = num
                    latest_snapshot = file
    # print("latest_snapshot: " + latest_snapshot)
    return latest_snapshot

def restore_train(net_name):
    latest_snapshot = snapshot_dir + get_latest_snapshot(net_name)
    print("restore training from latest snapshot {0}".format(latest_snapshot))

    # caffe train -solver alexnet_solver.prototxt -snapshot snapshot/alexnet_iter_1286.solverstate
    cmd = "caffe train --solver={0}_solver.prototxt --snapshot={1} 2>&1|tee {0}_train.log".format(net_name, latest_snapshot)
    print(cmd)
    os.system(cmd)

def finetuning(net_name, weights):
    print("start finetuning...")
    cmd = "caffe train --solver={0}_solver.prototxt --weights={1} 2>&1|tee {0}_train.log".format(net_name, weights)
    print(cmd)
    os.system(cmd)

if __name__ == "__main__":
    net_name = "cifar10_resnet20"

    import argparse
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-net", "--net", help='input net_name')
    parser.add_argument("-r", "--r", action='store_true', default=False,
                        help='Optional; restore training from latest snapshot.')
    parser.add_argument("-w", "--w",
                        help='Optional; the pretrained weights to initialize finetuning.')

    args = parser.parse_args()

    if args.net:
        net_name = args.net

    if not args.r:
        if not args.w:
            train_cmd(net_name)
        else:
            finetuning(net_name, args.w)
    else:
        restore_train(net_name)
