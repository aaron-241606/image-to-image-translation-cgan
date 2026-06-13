# Task-04: Image-to-Image Translation with cGAN (pix2pix)

> Implement an image-to-image translation model using a conditional generative adversarial network (cGAN) called pix2pix.

---

## 📌 Overview

This project implements the **pix2pix** architecture — a conditional GAN that learns a mapping from input images to output images. Example use cases include:
- Sketch → Photo
- Grayscale → Color
- Satellite → Map
- Day → Night

---

## 🗂️ Project Structure

```
image-to-image-translation-cgan/
├── model.py          # Generator (U-Net) + Discriminator (PatchGAN)
├── train.py          # Training loop with GAN losses
├── predict.py        # Translate a single input image
├── dataset.py        # Paired image dataset loader
├── utils.py          # Helper functions (save images, plot losses)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/aaron-241606/image-to-image-translation-cgan.git
cd image-to-image-translation-cgan
pip install -r requirements.txt
```

### Download a pix2pix dataset
```bash
# Example: facades dataset
wget http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz
tar -xzf facades.tar.gz
```

---

## 🚀 Usage

### Train the model
```bash
python train.py --data_dir ./facades --epochs 200 --output ./checkpoints
```

### Translate a single image
```bash
python predict.py --model ./checkpoints/generator_epoch200.pth --input input.jpg --output result.jpg
```

---

## 🧠 Architecture

```
Generator  : U-Net (Encoder-Decoder with skip connections)
             Input(256×256) → 8 encoder blocks → 8 decoder blocks → Output(256×256)

Discriminator : PatchGAN
             Classifies 70×70 overlapping patches as real/fake
             More effective than full-image discrimination
```

| Component      | Details                          |
|----------------|----------------------------------|
| Generator      | U-Net with skip connections      |
| Discriminator  | 70×70 PatchGAN                   |
| Loss           | cGAN loss + L1 loss (λ=100)      |
| Optimizer      | Adam (lr=0.0002, β1=0.5)         |
| Input size     | 256×256                          |

---

## 📚 References

- [#1 Image-to-Image Translation with Conditional GANs (Isola et al., 2017)](https://arxiv.org/abs/1611.07004)
- [#2 pix2pix Official Implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [#3 PatchGAN Discriminator Explained](https://paperswithcode.com/method/patchgan)

---

## 🏢 Credits

**Prodigy Infotech** – Task-04 Internship Project
