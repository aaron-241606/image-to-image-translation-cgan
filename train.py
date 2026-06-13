"""
Task-04: pix2pix cGAN Training Script
Prodigy Infotech Internship
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from model import UNetGenerator, PatchGANDiscriminator, initialize_weights
from dataset import PairedImageDataset


def train(data_dir, output_dir, epochs, batch_size, lr, lambda_l1, img_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    dataset = PairedImageDataset(data_dir, transform=transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    print(f"📂 Dataset: {len(dataset)} pairs | {len(loader)} batches/epoch\n")

    G = UNetGenerator().to(device)
    D = PatchGANDiscriminator().to(device)
    initialize_weights(G)
    initialize_weights(D)

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    bce = nn.BCEWithLogitsLoss()
    l1  = nn.L1Loss()

    for epoch in range(1, epochs + 1):
        G.train(); D.train()
        g_losses, d_losses = [], []

        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)

            # ── Train Discriminator ──────────────────────────
            fake = G(src)
            D_real = D(src, tgt)
            D_fake = D(src, fake.detach())
            loss_D = (bce(D_real, torch.ones_like(D_real)) +
                      bce(D_fake, torch.zeros_like(D_fake))) * 0.5
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            # ── Train Generator ──────────────────────────────
            D_fake = D(src, fake)
            loss_G_gan = bce(D_fake, torch.ones_like(D_fake))
            loss_G_l1  = l1(fake, tgt) * lambda_l1
            loss_G     = loss_G_gan + loss_G_l1
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

            g_losses.append(loss_G.item())
            d_losses.append(loss_D.item())

        avg_g = sum(g_losses) / len(g_losses)
        avg_d = sum(d_losses) / len(d_losses)
        print(f"Epoch [{epoch:>3}/{epochs}] | G Loss: {avg_g:.4f} | D Loss: {avg_d:.4f}")

        if epoch % 10 == 0 or epoch == epochs:
            save_image(fake[:4] * 0.5 + 0.5,
                       os.path.join(output_dir, "samples", f"epoch_{epoch:03d}.png"), nrow=2)
            torch.save(G.state_dict(), os.path.join(output_dir, f"generator_epoch{epoch}.pth"))
            torch.save(D.state_dict(), os.path.join(output_dir, f"discriminator_epoch{epoch}.pth"))
            print(f"   💾 Checkpoint saved.")

    print(f"\n✅ Training complete! Models saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pix2pix cGAN")
    parser.add_argument("--data_dir",  type=str,   default="./facades",     help="Path to paired dataset")
    parser.add_argument("--output",    type=str,   default="./checkpoints", help="Output directory")
    parser.add_argument("--epochs",    type=int,   default=200,             help="Number of training epochs")
    parser.add_argument("--batch",     type=int,   default=4,               help="Batch size")
    parser.add_argument("--lr",        type=float, default=0.0002,          help="Learning rate")
    parser.add_argument("--lambda_l1", type=float, default=100.0,           help="L1 loss weight")
    parser.add_argument("--img_size",  type=int,   default=256,             help="Image size")
    args = parser.parse_args()
    train(args.data_dir, args.output, args.epochs, args.batch, args.lr, args.lambda_l1, args.img_size)
