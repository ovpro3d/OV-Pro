<!-- <img src="docs/open_mmlab.png" align="right" width="30%"> -->

# OV-Pro: Enhancing open-vocabulary 3D object detection by prototype contrastive distillation

Official repository for OV-Pro: Enhancing open-vocabulary 3D object detection by prototype contrastive distillation.


**Highlights**: 
* `OV-Pro` has been submitted to PR and is currently under major revision.

## Overview
- [Installation](docs/INSTALL.md)
- [Quick Demo](docs/DEMO.md)
- [Getting Started](docs/GETTING_STARTED.md)
- [Citation](#citation)


## Introduction
OV-Pro is a novel framework built upon prototype-based knowledge distillation, designed to improve the model’s localization and classification capabilities for novel objects. Specifically, OV-Pro first uses the cross-view instance consistency constrained prototype generation (C3PG) method to construct open-vocabulary prototypes with more comprehensive geometric representations. This reduces noise in pseudo labels while enhancing the quality of prototype generation. Secondly, inspired by human capability to recognize incomplete objects, OV-Pro adopts the prototype contrastive distillation (PCD) method to transfer prototype knowledge at the instance level, enhancing the model’s recognition ability for incomplete novel objects. Finally, OV-Pro leverages the prototype propagation strategy (PPS) to propagate the prototype knowledge and simulate hard instances, improving the model’s robustness for novel objects. Extensive experiments on the nuScenes dataset demonstrated that the proposed framework achieves state-of-the-art performance in OV-3DDet tasks.

<p align="center">
  <img src="docs/Image.svg" width="90%" alt="OV-Pro Framework">
</p>

## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for the installation of `OpenPCDet`.

## Getting Started

Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) to get started with `OpenPCDet`.

### VLM Predictions
Download [nuscenes_infos_train_mono3d.coco.json](https://pan.baidu.com/s/19MEpBIB_1YePcc7OCQGZ8A?pwd=kvvi) and [nuscenes_glip_train_pred.pth](https://pan.baidu.com/s/19MEpBIB_1YePcc7OCQGZ8A?pwd=kvvi) to OV-Pro/data/training_pred. This will be loaded by the PreprocessedGLIP class in pcdet/models/preprocessed_detector.py to generate pseudo-labels.

### Training Process

1.  Extract `C3PG` Boxes and Prototypes

```
python tools/generate_prototype_pseudo_labels.py --cfg_file cfgs/nuscenes_cross_view_instance_consistency.yaml --folder ../data/pseudo_labels/nuscenes_prototype&box_proposals/
```

2. Run `PCD`

```
python train_st.py  --cfg_file tools/cfgs/OV_Pro_transfusion_unknown_x.yaml
```

3. Evaluate on all classes

```
python test.py --cfg_file cfgs/nuscenes_models/transfusion_lidar.yaml --ckpt ../output/OV_Pro_transfusion_unknown_x/default/ckpt/checkpoint_epoch_30.pth
```



## License

`OpenPCDet` is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement
`OpenPCDet` is an open source project for LiDAR-based 3D scene perception that supports multiple
LiDAR-based perception models as shown above. Some parts of `PCDet` are learned from the official released codes of the above supported methods. 
We would like to thank for their proposed methods and the official implementation.   

We hope that this repo could serve as a strong and flexible codebase to benefit the research community by speeding up the process of reimplementing previous works and/or developing new methods.


## Citation 
If you find this project useful in your research, please consider citing:


```
@misc{openpcdet2020,
    title={OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Clouds},
    author={OpenPCDet Development Team},
    howpublished = {\url{https://github.com/open-mmlab/OpenPCDet}},
    year={2020}
}
```


