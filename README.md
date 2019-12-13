# Joint-task Self-supervised Learning for Temporal Correspondence

[**Project**](https://sites.google.com/view/uvc2019) | [**Paper**]()

# Overview

![](docs/teaser.png)

[Joint-task Self-supervised Learning for Temporal Correspondence]()

[Xueting Li*](https://sunshineatnoon.github.io/), [Sifei Liu*](https://www.sifeiliu.net/), [Shalini De Mello](https://research.nvidia.com/person/shalini-gupta), [Xiaolong Wang](https://www.cs.cmu.edu/~xiaolonw/), [Jan Kautz](http://jankautz.com/), [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/).

(* equal contributions)

In  Neural Information Processing Systems (NeurIPS), 2019.

# Citation
If you use our code in your research, please use the following BibTex:

```
@inproceedings{uvc_2019,
    Author = {Xueting Li and Sifei Liu and Shalini De Mello and Xiaolong Wang and Jan Kautz and Ming-Hsuan Yang},
    Title = {Joint-task Self-supervised Learning for Temporal Correspondence},
    Booktitle = {NeurIPS},
    Year = {2019},
}
```
# Test on JHMDB
Run:
```
python test_jhmdb.py --use_softmax True --use_l2norm True --evaluate --topk_vis 20 --sigma 0.5 --cropSize 320 --cropSize2 80
```

# Acknowledgements
- This code is based on [TPN](https://arxiv.org/pdf/1804.08758.pdf) and [TimeCycle](https://github.com/xiaolonw/TimeCycle).
- For any issues, please contact xli75@ucmerced.edu or sifeil@nvidia.com.
