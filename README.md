Download alexnet weights:

```
wget http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy
```

Put the weights into the same directory as the git.

Then run `dump_pb.py` 

```
python3 dump_pb.py
```

and a frozen protobuf will be saved to alexnex\_frozen.pb

Download saved alexnet.pb at:

[alexnet\_frozen.pb](http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/alexnet_frozen.pb "alexnet_frozen.pb")

