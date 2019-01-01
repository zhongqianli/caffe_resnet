from caffe.proto import caffe_pb2

# net: "models/bvlc_alexnet/train_val.prototxt"
# test_iter: 1000
# test_interval: 1000
# base_lr: 0.01
# lr_policy: "step"
# gamma: 0.1
# stepsize: 100000
# display: 20
# max_iter: 450000
# momentum: 0.9
# weight_decay: 0.0005
# snapshot: 10000
# snapshot_prefix: "models/bvlc_alexnet/caffe_alexnet_train"
# solver_mode: GPU

import os
snapshot_dir = "snapshot/"
if not os.path.exists(snapshot_dir):
    os.mkdir(snapshot_dir)

def create_solver_step(net_name):
    s = caffe_pb2.SolverParameter()
    s.net = "{0}_train_test.prototxt".format(net_name)

    s.test_interval = 500
    s.test_iter.append(100)

    s.base_lr = 0.01
    s.momentum = 0.9
    s.weight_decay = 0.0005

    s.lr_policy = "step"
    s.gamma = 0.96
    s.stepsize = 5000

    s.display = 100

    s.max_iter = 20000

    s.snapshot = 1000

    s.snapshot_prefix = "{0}{1}".format(snapshot_dir, net_name)

    s.type = "SGD"

    s.solver_mode = caffe_pb2.SolverParameter.GPU

    filename = "{0}_solver.prototxt".format(net_name)
    with open(filename, "w") as f:
        f.write(str(s))

def create_solver_multistep(net_name):
    s = caffe_pb2.SolverParameter()
    s.net = "{0}_train_test.prototxt".format(net_name)

    s.test_interval = 1000
    s.test_iter.append(100)

    s.base_lr = 0.1
    s.momentum = 0.9
    s.weight_decay = 0.0001

    s.lr_policy = "multistep"
    s.gamma = 0.1

    s.display = 100

    s.max_iter = 64000

    s.snapshot = 10000

    s.snapshot_prefix = "{0}{1}".format(snapshot_dir, net_name)

    s.type = "SGD"

    s.solver_mode = caffe_pb2.SolverParameter.GPU

    filename = "{0}_solver.prototxt".format(net_name)
    with open(filename, "w") as f:
        f.write(str(s))

    lines = []
    with open(filename, "r") as f:
        for line in f:
            lines.append(line)

    # define stepvalue
    stepvalues = [32000, 48000]
    s = ""
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            if not line.find("multistep") == -1:
                for j in range(len(stepvalues)):
                    lines.insert(i+1+j, "stepvalue:{0}\n".format(stepvalues[j]))

        s = s.join(lines)

    with open(filename, "w") as f:
        f.write(s)


if __name__ == "__main__":
    # create_solver_step("resnet50")

    create_solver_multistep("cifar10_resnet20")
    create_solver_multistep("cifar10_resnet32")
    create_solver_multistep("cifar10_resnet44")
    create_solver_multistep("cifar10_resnet56")
    create_solver_multistep("cifar10_resnet110")