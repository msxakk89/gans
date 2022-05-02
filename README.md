# gans

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

Train Generative Adversarial Networks and monitor their evolution: Keras CPU implementation and datasets (Tensorflow backend)

---

## Quick Start

1. Install requirements
2. Run `gan_tainer.py` program
3. Inspect generated images

## Requirements

- Python        3.7.13
- Tensorflow    2.8.0
- Keras         2.8.0
- Numpy         1.21.6
- Matplotlib    3.2.2
- Imageio       2.4.1

## Usage

Use `-h` to see usage of `gan_tainer.py`:

```
usage: gan_trainer.py [-h] [--d D] [--epochs EPOCHS] [--batch_size BATCH] [--interval INTERVAL]
                      [--r R] [--c C] [--path PATH] [--anim_file ANIM_FILE]

Train Generative Adversarial Network (GAN) using Keras datasets

optional arguments:
  -h, --help            show this help message and exit
  --d D                 Select Keras pictures data to train GAN. Choose from 'mnist' and
                        'fashion_mnist'. Defulats to 'mnist' else 'fashion_mnist' chosen
  --epochs EPOCHS       Define number of epochs for GAN training. Defaults to 5000
  --batch_size BATCH    Define batch size. Defaults to 64
  --interval INTERVAL   Define interval after which generated images are drawn and training stats are
                        displayed. Defaults to 200
  --r R                 Number of rows in generated image. Defaults to 5
  --c C                 Number of columns in generated image. Defaults to 5
  --path PATH           Folder name or full path where drawn images will be saved. Defaults to
                        'images'
  --anim_file ANIM_FILE
                        Name of GIF animation file. Should end of .gif. Defaults to 'anim.gif'
```

## Example usage

Within terminal change to the cloned directory and run:

`python gan_trainer.py`

This will run the GAN training procedure with default parameters. Folder `/images` will be generated and contain generated images as the generator improves. An animation `anim.gif` will show generator evolution. 
