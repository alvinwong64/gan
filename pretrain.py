import argparse
import os
import numpy as np
import sys
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model import *
from dataset import *
from loss import *
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary 
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter  # TensorBoard import

if __name__ == "__main__": 


    experiment_name = f"experiment_ori_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    log_dir = os.path.join("logs", experiment_name)
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=400, help="num of epoch")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--hr_height", type=int, default=96, help="high res. image height")
    parser.add_argument("--hr_width", type=int, default=96, help="high res. image width")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cuda = torch.cuda.is_available()

    hr_shape = (args.hr_height, args.hr_width)
    
    dataloader = DataLoader(
        TrainImageDataset(r"D:\alvin\gan\DIV2K_train_HR\DIV2K_train_HR", hr_shape=hr_shape),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    testload= DataLoader(
        TestImageDataset(r"D:\alvin\gan\DIV2K_valid_HR\DIV2K_valid_HR", 4),
        batch_size=1,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    # Initialize generator and discriminator
    generator = SRGenerator()
    # discriminator = Discriminator()


    # generator.load_state_dict(torch.load("saved_models/generator_50.pth"))

    # Losses
    criterion_mse = nn.MSELoss()
    criterion_adv = AdversarialLoss()
    criterion_BCE = nn.BCELoss()
    criterion_content = ContentLoss()

    if cuda:
        generator = generator.cuda()
        # discriminator = discriminator.cuda()    
        criterion_adv = criterion_adv.cuda()
        criterion_content = criterion_content.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    # optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # ---------- Training ----------
    for epoch in range(args.epoch):
        train_bar = tqdm(dataloader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd(X)': 0, 'dg(x1)': 0, 'dg(x2)':0}
        # if epoch == 100:
        #     discriminator = Discriminator()
        #     # discriminator = discriminator.cuda()
        #     optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
    


        generator.train()
        # discriminator.train()

        for i, imgs in enumerate(train_bar):

            # Configure model input
            imgs_lr = imgs["lr"].to(device)
            imgs_hr = imgs["hr"].to(device)

            batch_size = imgs_lr.size(0)
            real_label = torch.full([batch_size, 1], 1.0, dtype=torch.float, device=device,requires_grad=False)
            fake_label = torch.full([batch_size, 1], 0.0, dtype=torch.float, device=device,requires_grad=False)
            running_results['batch_sizes'] += batch_size

            # ------------------ Train Generator ------------------
            optimizer_G.zero_grad()

            # # Generate high resolution image from low resolution input
            gen_hr = generator(imgs_lr)

            # if epoch < 100 :
            loss_G = F.mse_loss(gen_hr,imgs_hr)
            
            # else:

            # # Adversarial loss
            #     fake_probs = discriminator(gen_hr)
            #     loss_GAN = criterion_adv(fake_probs)

            #     # Content loss
            #     loss_content = criterion_content(gen_hr, imgs_hr)

            #     # Total loss for generator
            #     loss_G = loss_content + 1e-3 * loss_GAN
            # # loss_G = F.mse_loss(gen_hr,imgs_hr)

            loss_G.backward()
            optimizer_G.step()
        #      # ----------------- Train Discriminator -----------------
            
        #         # Real images
        #     real_probs = discriminator(imgs_hr)
        #     loss_real = criterion_BCE(real_probs, real_label)
        #     D_x = real_probs.mean().item()
            
        #     # Fake images
        #     fake_probs = discriminator(gen_hr.detach())
        #     loss_fake = criterion_BCE(fake_probs, fake_label)
        #     D_G_x1 = fake_probs.mean().item()
            
        #     # Combined loss
        #     loss_D = loss_real+loss_fake
            

        #     optimizer_D.zero_grad()
        #     loss_D.backward()
        #     optimizer_D.step()
            
        #     D_G_x2 = fake_probs.mean().item()
        #     save_image(imgs_lr, f"images/lr/{epoch}.png" , normalize=False)




        #     # -------------- Log Progress Every Iteration --------------
        #     running_results['g_loss'] += loss_G.item() *batch_size
        #     running_results['d_loss'] += loss_D.item() *batch_size
        #     running_results['d(X)'] += D_x *batch_size
        #     running_results['dg(x1)'] += D_G_x1 *batch_size
        #     running_results['dg(x2)'] += D_G_x2 *batch_size


        #     # Log scalar values to TensorBoard at each iteration
        #     writer.add_scalar('Loss/Generator', loss_G.item(), epoch * len(dataloader) + i)
        #     writer.add_scalar('Loss/Discriminator', loss_D.item(), epoch * len(dataloader) + i)
        #     writer.add_scalar('D(x)', D_x, epoch * len(dataloader) + i)
        #     writer.add_scalar('D(G(z))', D_G_x2, epoch * len(dataloader) + i)

        #     train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (
        #         epoch, args.epoch, running_results['d_loss'] / running_results['batch_sizes'],
        #         running_results['g_loss'] / running_results['batch_sizes'],
        #         running_results['d(X)'] / running_results['batch_sizes'],
        #         running_results['dg(x1)'] / running_results['batch_sizes'],
        #         running_results['dg(x2)'] / running_results['batch_sizes'] ))

        os.makedirs(f"images/{experiment_name}", exist_ok=True)
        os.makedirs(f"saved_models/{experiment_name}", exist_ok=True)
        # # -------- Log After Every Epoch --------
        # avg_d_loss = running_results['d_loss'] / running_results['batch_sizes']
        # avg_g_loss = running_results['g_loss'] / running_results['batch_sizes']
        # avg_d_score = running_results['d(X)'] / running_results['batch_sizes']
        # avg_g_score = running_results['dg(x2)'] / running_results['batch_sizes']

        # # Log scalar values at the end of the epoch
        # writer.add_scalar('Loss/Generator_Epoch', avg_g_loss, epoch)
        # writer.add_scalar('Loss/Discriminator_Epoch', avg_d_loss, epoch)
        # writer.add_scalar('D(x)_Epoch', avg_d_score, epoch)
        # writer.add_scalar('D(G(z))_Epoch', avg_g_score, epoch)
        
        # # image_grids = []
        # idx = 0
        # torch.cuda.empty_cache()  # Clear any cached memory

        # Testing phase
        with torch.no_grad():  # D
            for lr, sr, hr in tqdm(testload):
                lr = lr.cuda()
                sr = sr.cuda()
                hr = hr.cuda()

                gen_hr = generator(lr)
                imgs_lr_resized = sr
                gen_hr_grid = make_grid(gen_hr, nrow=1, normalize=True)
                img_hr_grid = make_grid(hr.cuda(), nrow=1, normalize=True)
                imgs_lr_grid = make_grid(imgs_lr_resized, nrow=1, normalize=True)

                # Concatenate LR, generated HR, and true HR images horizontally
                img_grid = torch.cat((imgs_lr_grid, gen_hr_grid, img_hr_grid), dim=-1)


                # save_image(img_grid, f"images/{experiment_name}/{epoch}.png", normalize=False)
                writer.add_image('Generated Images', img_grid, epoch)
                break


        # Save model checkpoints every 10 epochs
        if epoch % 10 == 0 or epoch == args.epoch-1:
            torch.save(generator.state_dict(), f"saved_models/{experiment_name}/generator_{epoch}.pth")
            # torch.save(discriminator.state_dict(), f"saved_models/{experiment_name}/discriminator_{epoch}.pth")

    # Close the writer
    writer.close()
