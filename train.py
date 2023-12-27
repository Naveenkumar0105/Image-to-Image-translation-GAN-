import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset_loading import MapDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm #library for displaying progress bars
from torchvision.utils import save_image


def train_fn(disc, gen, loader, opt_disc, opt_gen, l1, bce, g_scaler, d_scaler):
    loop = tqdm(loader, leave=True) # sets up the progress bar to monitor the training process

    for idx, (x, y) in enumerate(loop):
        # moving the input and target images to gpu if available or else CPU
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast(): # enables automatic mixed precision (AMP) training in PyTorch.
            # It involves using lower-precision data types for certain operations to speed up training
            # while maintaining the necessary precision in critical parts of the network.
            y_fake = gen(x) # generates fake images
            D_real = disc(x, y) # discriminator's output for the real input and output images
            # since the D_real actually had to have only 1's since the output was the actual correct image,
            # we are checking whether it was predicted correctly by the discriminator, or else
            # it is considered as a loss
            # torch.ones_like(D_real) creates a tensor of the same shape as D_real and fills it with 1's
            D_real_loss = bce(D_real, torch.ones_like(D_real))# we are finding the binary cross entropy loss
            D_fake = disc(x, y_fake.detach()) # disconnects a tensor from the computational graph
            # detaching the generated images prevents the gradients from flowing back to the generator,
            # ensuring independent training, but it brings all the data from the image generated
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2 # the final discriminator loss is taken as the average of
            # losses on the real and fake images. This is a common practice in GAN training. It represents the
            # performance of the discriminator on the entire batch

        disc.zero_grad() # zero out the gradients of the discriminator's parameters. It is necessary before
        # computing the gradients in the backward pass
        d_scaler.scale(D_loss).backward() #Scales the loss to avoid potential numerical stability issues
        # that may arise (underflow or overflow) when training with lower precision data types
        # backward pass is done to upgrade the gradients (part of the optimization process)
        d_scaler.step(opt_disc) # updates the discriminator's parameters based on the scaled gradients
        d_scaler.update() # updating the gradient scaler for the next iteration

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1(y_fake, y) * config.L1_LAMBDA
            # config.L1_LAMBDA is a regularization term to encourage similarity between the generated
            # and the target image. It is used to scale the L1 loss
            G_loss = G_fake_loss + L1 # the generator aims to minimize this loss during training

        # same steps as the discriminator part as we can see
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()


def main():
    # initialization
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999)) # (beta1, beta2)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    # if the model is already present, then load the model
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)

    train_dataset = MapDataset(root_dir="archive/maps/maps/train")
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    # DataLoader helps us to efficiently load and iterate over batches of data
    # the higher the value of num_workers, the more parallelism can be done during data loading
    # creating a gradient scaler to handle mixed precision training
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    # setting up the validation dataset
    val_dataset = MapDataset(root_dir="archive/maps/maps/val")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler)

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        save_some_examples(gen, val_loader, epoch, folder="evaluation")


if __name__ == "__main__":
    main()