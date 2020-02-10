## Future Person Localization in First-Person Videos (CVPR2018)
<img src="https://github.com/takumayagi/fpl/blob/image/cvpr18_teaser.png" width="80%" height="80%">

This repository contains the code and data (**caution: no raw image provided**) for the paper ["Future Person Localization in First-Person Videos"](https://arxiv.org/abs/1711.11217) by Takuma Yagi, Karttikeya Mangalam, Ryo Yonetani and Yoichi Sato.

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
[Download link (processed data)](https://drive.google.com/file/d/1Px3f3_oJqzJT10TQfEPvxPvSllEF2LSz/view?usp=sharing)

If you wish downloading via terminal, consider using [custom script](https://gist.github.com/darencard/079246e43e3c4b97e373873c6c9a3798).

Extract the downloaded tar.gz file at the root directory.
```
tar xvf fpl.tar.gz
```

### Pseudo-video
Since we cannot release the raw images, we prepared sample pseudo-video below.  
The video shows the automatically extracted location histories, poses. The number shown in the bounding box corresponds to the person id in the processed data.    
Background colors are the result from pre-trained dilated CNN trained with [MIT Scene Parsing Benchmark](http://sceneparsing.csail.mit.edu/).
<img src="https://github.com/takumayagi/fpl/blob/image/ezgif-1-9c3c383428.gif">  
[Download link (pseudo-video)](https://drive.google.com/file/d/1mYIth2npDULSquVkYMbownMy7xJ8Tk1H/view?usp=sharing)

### Create dataset
Run dataset generation script to preprocess raw locations/poses/egomotions.  
A single processed file will be generated in datasets/.
```
# Test (debug) data
python utils/create_dataset.py utils/id_test.txt --traj_length 20 --traj_skip 2 --nb_splits 5 --seed 1701 --traj_skip_test 5
# All data
python utils/create_dataset.py utils/id_list_20.txt --traj_length 20 --traj_skip 2 --nb_splits 5 --seed 1701 --traj_skip_test 5
```

### Prepare training script
Modify the "in_data" arguments in scripts/5fold.json.

## Running the code
### Directory structure
```
    .
    +---data (feature files)
    +---dataset (processed data)
    +---experiments (logging)
    +---gen_scripts (automatically generated scripts for cross validation)
    +---models
    +---scripts (configuration)
    |   +---5fold.json
    +---utils
        +---run.py (training script)
        +---eval.py (evaluation script)
```

### Training
In our environment (a single TITAN X Pascal w/ CUDA 8, cuDNN 5.1), it took approximately 40 minutes per split.
```
# Train proposed model and ablation models
python utils/run.py scripts/5fold.json run <gpu id>
# Train proposed model only
python utils/run.py scripts/5fold_proposed_only.json run <gpu id>
```

### Evaluation
```
python utils/eval.py experiments/5fold_yymmss_HHMMSS/ 17000 run <gpu id> 10
```

### Prediction visualization using pseudo-video
We provided visualization code using pseudo-video.  
Download below pseudo-videos and run the following code:  
[Download link (pseudo-videos for visualization)](https://drive.google.com/open?id=1akSmtlfIbEwlsy85iYb7s1U9e0zBeP1T)  

```
# Run this code after placing <video_id>.mp4 into data/pseudo_viz/
# Extract images from video
python utils/video2img_all.py data/pseudo_viz/
# Plot images
python utils/plot_prediction.py <experiment>/<fold> --traj_type 0
# Write videos
python utils/write_video.py <experiment>/<fold> --vid GOPRXXXXU20 --frame XXXX --pid XXX
```

## License and Citation
The dataset provided in this repository is only to be used for non-commercial scientific purposes. If you used this dataset in scientific publication, cite the following paper:

Takuma Yagi, Karttikeya Mangalam, Ryo Yonetani and Yoichi Sato. Future Person Localization in First-Person Videos. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
```
@InProceedings{yagi2018future,
    title={Future Person Localization in First-Person Videos},
    author={Yagi, Takuma and Mangalam, Karttikeya and Yonetani, Ryo and Sato, Yoichi},
    booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2018}
}
```
