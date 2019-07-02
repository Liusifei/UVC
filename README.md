# Joint-task Self-supervised Learning for Temporal Correspondence

This code is based on https://gitlab-master.nvidia.com/sifeil/pytorch_tpn

# One time setup

```
git clone <this repo>  
cd uvc
```

## Docker image

Please use ```nvcr.io/nvidian_general/sifeil:pytorch0.5_ply``` based on Pytorch 0.5.

## Dataset

We use the [kinetics dataset](https://deepmind.com/research/open-source/open-source-datasets/kinetics/).

cosmos: ```//dcg-zfs-03.nvidia.com/nvidia_data.cosmos665```

NGC: ```$ ngc dataset download 24552```

## Train

```
python track_match_v1.py --wepoch 10 --nepoch 30 -c match_track_switch --batchsize 40 --coord_switch 0 --lc 0.3
```