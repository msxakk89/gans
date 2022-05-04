# gans

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

Train Generative Adversarial Networks and monitor their evolution: Keras implementation and datasets (Tensorflow backend)

---

## Quick Start

1. Install requirements
2. Run `gan_tainer.py` program
3. Inspect images

### Example

Within terminal change to the cloned directory and run:

`python gan_trainer.py`

This will run the GAN training procedure with default parameters. Folder `/images` will be created with generated images at different stages of GAN training. 

## Requirements

Install the following:

- Python        3.7.13
- Tensorflow    2.8.0
- Keras         2.8.0
- Numpy         1.21.6
- Matplotlib    3.2.2
- Imageio       2.4.1

Altenatively you can use the [following Docker container](https://www.dropbox.com/s/lsbgl03v15at37g/gan.tar?dl=0)

## Usage

Use `-h` to see usage of `gan_tainer.py`:

```
usage: gan_trainer.py [-h] [--d D] [--epochs EPOCHS] [--batch_size BATCH]
                      [--interval INTERVAL] [--r R] [--c C] [--path PATH]
                      [--mod_path MOD_PATH] [--train_on_number TRAIN_ON_NUMBER]
                      [--train_on_subset TRAIN_ON_SUBSET] [--create_anim CREATE_ANIM]
                      [--anim_file ANIM_FILE]

Train Generative Adversarial Network (GAN) using Keras datasets

optional arguments:
  -h, --help            show this help message and exit
  --d D                 Select Keras pictures data to train GAN. Choose from 'mnist' and
                        'fashion_mnist'. Defulats to 'mnist' else 'fashion_mnist' chosen
  --epochs EPOCHS       Define number of epochs for GAN training. Defaults to 5000
  --batch_size BATCH    Define batch size. Defaults to 64
  --interval INTERVAL   Define interval after which generated images are drawn and training
                        stats are displayed. Defaults to 200
  --r R                 Number of rows in generated image. Defaults to 5
  --c C                 Number of columns in generated image. Defaults to 5
  --path PATH           Folder name or full path where drawn images will be saved. Defaults
                        to 'images'
  --mod_path MOD_PATH   Folder name or full path where models will be saved. Defaults to
                        'models'
  --train_on_number TRAIN_ON_NUMBER
                        If 'mnist' data selected train on a subset of this number only. Must
                        be an integer between 0 and 9. Defaults to -1 which implies all
                        numbers
  --train_on_subset TRAIN_ON_SUBSET
                        If 'fashion_mnist' data selected train on a subset of this number
                        only. Must be an integer between 0 and 9. See
                        [https://keras.io/api/datasets/fashion_mnist/] for the explanation
                        of numbers. Defaults to -1 which implies all types
  --create_anim CREATE_ANIM
                        Should a GIF animation be created? Defaults to False
  --anim_file ANIM_FILE
                        Name of GIF animation file. Should end with .gif. Defaults to
                        'anim.gif'
```


