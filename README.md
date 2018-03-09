Pretrained AlexNet model for TensorFlow
--------------------

This repository is a fork of [kratzert/finetune_alexnet_with_tensorflow](https://github.com/kratzert/finetune_alexnet_with_tensorflow/),
and have been adapted to generate a frozen protobuf for AlexNet.

First download alexnet weights (from caffee) in .npy format:

```
wget http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy
```

Put the weights into the same directory as the this git repository.

Then run `dump_pb.py` 

```
python3 dump_pb.py
```

Then a network with trainable weights is saved to alexnet.pb, and a frozen protobuf is saved to alexnex\_frozen.pb

Download saved alexnet.pb and alexnet_frozen.pb here:

[alexnet.pb](http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/alexnet.pb "alexnet.pb")

[alexnet\_frozen.pb](http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/alexnet_frozen.pb "alexnet_frozen.pb")

