# future\_localization: Future localization from second-person view
This repository provides future localization of person in the scene.

## Requirements

- GPU environment (recommended)
- Python 2.7.6+, 3.4.3+, 3.5.1+
- [Chainer](https://github.com/pfnet/chainer) v2.0.0
- [ChainerCV](https://github.com/chainer/chainercv) v0.5.1
- NumPy 1.9, 1.10, 1.11
- Cython 0.25+
- OpenCV 3.1+
- joblib
- mllogger (https://github.com/Signull8192/mllogger)
- six
- pandas
- numpy-quaternion
- numba
- python-box

## Main features

- RNN-ED baseline
- RNN-ED with Scene feature
- Simple RNN (SRNN) with pose/egomotion/mean flow feature

## Installation

### Setup
```
export TRAJ_DATA_DIR=<data directory> >> ~/.bashrc
```

In titanx3, please run commands as follows:
```
mkdir -p ~/data/
ln -s /ssd/yagi/data/ ~/data/videos  # GOPRXXXX folder should be in "videos" folder, but I forgot to do that
export TRAJ_DATA_DIR=~/data >> ~/.bashrc
```

## Directory structure

    .
    +---external
    +---rnn_ed
    |   +---(build)
    |   +---src
    |   +--CMakeLists.txt
    |   +---python
    +---rnn_ed_scene
    +---direct (current main directory)
    |   +---outputs (normal experiment output)
    |   +---experiments (when using run.py)
    +---utils
    
## How to run
### Dataset preparation
Ask Yagi to get Dataset file (includes input/output feature, without image data)

### Training
#### Train indivisually (result will be written in output/ directory)
```
cd direct/
python train.py --in_data dataset_video/front_id_list_10_171003_012633_20.joblib --nb_train 1000 --nb_iters 100 --height 960 --gpu 0 --iter_snapshot 50 --debug --nb_units 16  --model srnn
```

#### Train by script
First, edit script file to specify conditions to run
```
vim script/test_script.json
```

Script example:
```
{
    "script_name": "train.py",
    "comment": "Check effect of dataset size",
    "fixed_args": {
        "in_data": "dataset_video/id_list_10_171003_012437_20.joblib",
        "nb_iters": 30000,
        "iter_snapshot": 3000,
        "gpu": 0,
        "unit_type": "NStepGRU",
        "rnn_dropout": 0.3,
        "nb_layers": 2,
        "optimizer": "adam",
        "height": 960,
        "batch_size": 32,
        "save_model": "",
        "nb_units": 32,
        "model": "srnn"
    },
    "dynamic_args": {
        "nb_train": [1000, 2000, 5000, 10000, 20000, -1]
    }
}
```

Test script (to check the contents of the generated commands)
```
python run.py scripts/size_test.json test
```

```
python -u train.py --root_dir experiments/size_test_171004_232619 --optimizer adam --nb_iters 30000 --save_model  --iter_snapshot 3000 --rnn_dropout 0.3 --batch_size 32 --height 960 --unit_type NStepGRU --gpu 1 --model srnn --nb_units 32 --nb_layers 2 --in_data dataset_video/id_list_10_171003_012437_20.joblib --nb_train 1000 
python -u train.py --root_dir experiments/size_test_171004_232619 --optimizer adam --nb_iters 30000 --save_model  --iter_snapshot 3000 --rnn_dropout 0.3 --batch_size 32 --height 960 --unit_type NStepGRU --gpu 1 --model srnn --nb_units 32 --nb_layers 2 --in_data dataset_video/id_list_10_171003_012437_20.joblib --nb_train 2000 
python -u train.py --root_dir experiments/size_test_171004_232619 --optimizer adam --nb_iters 30000 --save_model  --iter_snapshot 3000 --rnn_dropout 0.3 --batch_size 32 --height 960 --unit_type NStepGRU --gpu 1 --model srnn --nb_units 32 --nb_layers 2 --in_data dataset_video/id_list_10_171003_012437_20.joblib --nb_train 5000 
python -u train.py --root_dir experiments/size_test_171004_232619 --optimizer adam --nb_iters 30000 --save_model  --iter_snapshot 3000 --rnn_dropout 0.3 --batch_size 32 --height 960 --unit_type NStepGRU --gpu 1 --model srnn --nb_units 32 --nb_layers 2 --in_data dataset_video/id_list_10_171003_012437_20.joblib --nb_train 10000 
python -u train.py --root_dir experiments/size_test_171004_232619 --optimizer adam --nb_iters 30000 --save_model  --iter_snapshot 3000 --rnn_dropout 0.3 --batch_size 32 --height 960 --unit_type NStepGRU --gpu 1 --model srnn --nb_units 32 --nb_layers 2 --in_data dataset_video/id_list_10_171003_012437_20.joblib --nb_train 20000 
python -u train.py --root_dir experiments/size_test_171004_232619 --optimizer adam --nb_iters 30000 --save_model  --iter_snapshot 3000 --rnn_dropout 0.3 --batch_size 32 --height 960 --unit_type NStepGRU --gpu 1 --model srnn --nb_units 32 --nb_layers 2 --in_data dataset_video/id_list_10_171003_012437_20.joblib --nb_train -1 
Test finished
```

Run 
```
python run.py scripts/size_test.json run
```
