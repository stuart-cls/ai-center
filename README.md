
aicenter
========

A python based Soft IOC Server.

Usage
=====
In order to use "aicenter", you need have a functioning install of python-devioc and its requirements and procServ.
 
1. Create a directory for the IOC instance. The directory should be named exactly like the device name but the location
   is irelevant. 
2. Copy the init-template file to /etc/init.d and rename it as appropriate.
3. Edit the file from (2) above to reflect your environment and to set all the required instance parameters
4. Enable the init file using your system commands. For example, `systemctl enable <init-file-name>`.
5. Start the init file using your system commands. For example `systemctl start <init-file-name>`.

You can manage the instance daemon through procServ, by telneting to the configured port. 

Installation
============

```
pip install .[ioc]
```

OpenCV
======

For best performance, a version of `python-opencv` compiled with support for CUDA and cuDNN along
with a compatible GPU should be used.

Testing
=======

The `test/inference.py` file can be used to test the inference / model performance without running
a full IOC application.

Install `aicenter` without `[ioc]` dependencies:

```
pip install .[test]
```

Segment Anything
================

To enable segmentation tracking with SAM2, install with `[sam]` extra.

Model weights must be downloaded from the [sam2](https://github.com/facebookresearch/sam2?tab=readme-ov-file#model-description)
page. Currently checkpoint files for SAM 2 (July 2024) are supported.

By default, the `aicenter.sam` module looks for weights in `<my-aicenter-venv>/sam_weights/sam2_hiera_large.pt`

### Acknowledgment
SAM support uses [muggled_sam](https://github.com/heyoeyo/muggled_sam) which itself is an
implementation of Segment Anything 2:

[facebookresearch/sam2](https://github.com/facebookresearch/sam2)
```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint},
  year={2024}
}
```
