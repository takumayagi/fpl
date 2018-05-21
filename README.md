## Future Person Localization in First-Person Videos (CVPR2018)
<img src="https://github.com/takumayagi/fpl/blob/image/cvpr18_teaser.png" width="80%" height="80%">

This repository contains the code and data (**caution: no raw image provided**) for the paper "Future Person Localization in First-Person Videos" by Takuma Yagi, Karttikeya Mangalam, Ryo Yonetani and Yoichi Sato.

## Prediction examples
<img src="https://github.com/takumayagi/fpl/blob/image/001.gif" width="20%" height="20%"><img src="https://github.com/takumayagi/fpl/blob/image/002.gif" width="20%" height="20%"><img src="https://github.com/takumayagi/fpl/blob/image/003.gif" width="20%" height="20%"><img src="https://github.com/takumayagi/fpl/blob/image/004.gif" width="20%" height="20%"><img src="https://github.com/takumayagi/fpl/blob/image/005.gif" width="20%" height="20%">

## Requirements
We confirmed the code works correctly in below versions.

- GPU environment
- Python 3.5.2
- [Chainer](https://github.com/pfnet/chainer) v4.0.0
- NumPy 1.13.1
- Cython 0.25.1
- OpenCV 3.3.0
- joblib 0.11
- mllogger (https://github.com/Signull8192/mllogger)
- pandas 0.20.3
- numpy-quaternion 2018.5.17.10.19.59
- numba 0.36.2
- python-box 3.2.0


## Installation
### Download data
You can download our dataset from below link:
**(caution: no raw image provided!)**
WIP

```
# WIP
```

### Create dataset
```
# Test data
python utils/create_dataset.py utils/id_test.txt --traj_length 20 --traj_skip 2 --nb_splits 5 --seed 1701 --traj_skip_test 5
# All data
python utils/create_dataset.py utils/id_list_20.txt --traj_length 20 --traj_skip 2 --nb_splits 5 --seed 1701 --traj_skip_test 5
```

### Prepare training script
Modify the "in_data" arguments in scripts/5fold.json.

## Training
```
python utils/run.py scripts/5fold.json run <gpu id>
```

## Evaluation
```
python utils/eval.py experiments/5fold_yymmss_HHMMSS/ 17000 run <gpu id> 10
```
