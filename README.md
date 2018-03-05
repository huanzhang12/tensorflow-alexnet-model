Download alexnet weights:

```
wget http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy
```

Put the weights into the same directory as the git.

Then run `dump_pb.py` 

```
python3 dump_pb.py
```

and protobuf will be saved to alexnex.pb

