# Knowledge Graph Attention Network
This is PyTorch & Pytorch Geometric implementation for the paper:
>Xiang Wang, Xiangnan He, Yixin Cao, Meng Liu and Tat-Seng Chua (2019). KGAT: Knowledge Graph Attention Network for Recommendation. [Paper in ACM DL](https://dl.acm.org/authorize.cfm?key=N688414) or [Paper in arXiv](https://arxiv.org/abs/1905.07854). In KDD'19, Anchorage, Alaska, USA, August 4-8, 2019.

You can find Tensorflow implementation by the paper authors [here](https://github.com/xiangwang1223/knowledge_graph_attention_network).

## Introduction
Knowledge Graph Attention Network (KGAT) is a new recommendation framework tailored to knowledge-aware personalized recommendation. Built upon the graph neural network framework, KGAT explicitly models the high-order relations in collaborative knowledge graph to provide better recommendation with item side information.

If you want to use codes and datasets in your research, please contact the paper authors and cite the following paper as the reference:
```
@inproceedings{KGAT19,
  author    = {Xiang Wang and
               Xiangnan He and
               Yixin Cao and
               Meng Liu and
               Tat{-}Seng Chua},
  title     = {{KGAT:} Knowledge Graph Attention Network for Recommendation},
  booktitle = {{KDD}},
  pages     = {950--958},
  year      = {2019}
}
```

## Environment Requirement
The code has been tested running under Python 3.6.9. The required packages are as follows:
* torch == 1.7.0
* numpy == 1.19.5
* scipy == 1.4.1
* sklearn == 0.22.2

For an installation guide of Pytorch Geometric see [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

## Run the Codes

In order to get similar [results](https://github.com/xiangwang1223/knowledge_graph_attention_network/blob/master/Log/training_log_last-fm.log) as the authors for the last-fm dataset you can run
* KGAT (Parameter/Bias Version) – around 1 minute faster
```
python main.py --cf_batch_size 10240 --ckg_batch_size 2048 --Ks [20,100]
```
* KGAT (Linear Layer Version) – around 1 minute faster
```
python main.py --model_type Linear --cf_batch_size 10240 --ckg_batch_size 2048 --Ks [20,100]
```
We achieve the similar results with around 30% time improvement using one Tesla P100.

## Further ToDo's
* Add SparseTensor support. This will further improve the speed of the method
* Add Logging support
* Add support for saving the run method, incl. early stopping
* Add AUC and hit@K to metrics.py


