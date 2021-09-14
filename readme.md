### Environment Requirement

The code runs well under python 3.6.5. The required packages are as follows:

* pytorch==1.6.0
* dgl==0.5.2
* numpy==1.14.3
* pytorch-ignite==0.4.2
* yaml==3.12
* shapely==1.7.1
* glob, pickle, argparse, matplotlib, pandas

### Datasets

The raw data is from Didi (https://outreach.didichuxing.com/research/opendata/）, and we have preprocessed these data into the format of our model's input, where the sampling data (all data is too big, 200G+😁) is provided in the following cloud disk (https://pan.baidu.com/s/1RfR82pzn9NCaKn3uN8NNEQ , the passcode is ej2c), and the default time slot size is 5 minutes and the past time slot length for each sample is 12. Hence, if you want to build your own dataset or other data with different slot size and past slot length, please feel free to contact me once the paper is accepted ☺.



### Run the code

The executed command is as follows:

python3 -u pipeline.py --device=cuda:2 --city=chengdu --yaml_file=config-bash.yaml --slot_size=5 --pre_slot_num=12 --split_scale=0.25 --global_num_hidden=64 --node_feature_dim=16 --local_num_hidden=32 --st_hidden_size=32 --st_total_key_depth=12 --st_total_value_depth=24 --day_embedding_dim=4 --time_embedding_dim=2 --train_batch_size=500 --test_batch_size=500 --epochs=20 --lr=0.001 --early_stop_epoch=5 --lamb=0.5

- device: cpu/cuda(0,1,2..), where cuda(0,1,...) means using GPU for training the model
- city: chengdu or xian, the default values is chengdu.
- yaml_file: the meta configure file name, whose location is under the directory "./code/".
- slot_size: the size of each time slot, the default value is set as 5 minutes.
- pre_slot_num: the length of past time slots, the default value is 12.
- split_scale: the rate of compact of speed histograms (raw length is 20). For example, if the default value is set as 0.25, and the size of each histogram would be 20*0.25 = 5.
- global_num_hidden: $d_{g_1}$
- node_feature_dim: $d_{g_2}$
- local_num_hidden: $d_{g'_1}$
- st_hidden_size: $d_{g'_2}$
- st_total_key_depth: $d_q$, $d_k$
- st_total_value_depth: $d_v$, $d_f$, $d_{hr}$
- day_embedding_dim: $d_w$
- time_embedding_dim: $d_d$
- train_batch_size: the batch size for training the model, the default value is 500
- test_batch_size: the batch size for testing the model, the default value is 500
- epochs: the maximum number of learning epochs, the default value is 50
- lr: the learning rate, the default value is 0.001
- early_stop_epoch: the number of epochs for early stop of training when there is no reduction of training loss.
- lamb: $\lambda$

