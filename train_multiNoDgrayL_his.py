#from lib
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.utils as nn_utils


#from files
import config
from utils import save_checkpoint, load_checkpoint, save_some_examples, save_gray_and_color_examples, final_save_all
from dataset import MapDataset, save_transformed_images, save_single_transformed
from dataset_multi import MapDataset_Multi
from generator import Generator
from discriminator import Discriminator
import os

torch.manual_seed(1000)
torch.autograd.set_detect_anomaly(True)



def train_loop(d1, g1, g2, loader, optimizer_d1, optimizer_g1, optimizer_g2, l1, bce, g1_scaler, g2_scaler, d1_scaler):
    #loop = tqdm(loader, leave=True)

    
    
    for idx, (x, y, y_gray) in enumerate(loader):
        x, y, y_gray = x.to(config.DEVICE), y.to(config.DEVICE), y_gray.to(config.DEVICE)
        
        # Train Discriminator
        with torch.cuda.amp.autocast(dtype=torch.float16):
            y_fake_gray = g1(x)
            y_fake_color = g2(y_fake_gray)
            #print(f"y_fake_gray version: {y_fake_gray._version}, y_fake_color version: {y_fake_color._version}")

            D_real_logits = d1(x, y)
            D_fake_logits_color = d1(x, y_fake_color.detach())
            D_fake_logits_gray = d1(x, y_fake_gray.detach())
            #print(f"D_real_logits version: {D_real_logits._version}, D_fake_logits_color version: {D_fake_logits_color._version}, D_fake_logits_gray version: {D_fake_logits_gray._version}")

            ##############################################
            D_real_logits = torch.clamp(D_real_logits, min=1e-7, max=1-1e-7)
            D_fake_logits_color = torch.clamp(D_fake_logits_color, min=1e-7, max=1-1e-7)
            D_fake_logits_gray = torch.clamp(D_fake_logits_gray, min=1e-7, max=1-1e-7)
            ################################################
            
            
            D_real_loss = bce(D_real_logits, torch.ones_like(D_real_logits))
            D_fake_loss_color = bce(D_fake_logits_color, torch.zeros_like(D_fake_logits_color))
            D_fake_loss_gray = bce(D_fake_logits_gray, torch.zeros_like(D_fake_logits_gray))

            D_loss = D_real_loss + D_fake_loss_color

        if torch.isnan(D_loss):
            print("NaN detected in D_loss")
            continue
        
        d1.zero_grad()
        d1_scaler.scale(D_loss).backward()
        
        
        
        
        nn_utils.clip_grad_norm_(d1.parameters(), max_norm=1.0)
        
        
        
        
        d1_scaler.step(optimizer_d1)
        d1_scaler.update()
        
        


        # Train Generator 1
        with torch.cuda.amp.autocast(dtype=torch.float16):
            D_fake_logits_gray = d1(x, y_fake_gray)
            #print(f"D_fake_logits_gray version before backward: {D_fake_logits_gray._version}")
            D_fake_logits_gray = torch.clamp(D_fake_logits_gray, min=1e-7, max=1-1e-7)
            ###################################################################
            
            
            G_fake_loss_gray = bce(D_fake_logits_gray, torch.ones_like(D_fake_logits_gray))
            L1_gray = l1(y_fake_gray, y_gray) * config.L1_LAMBDA
            G1_loss = G_fake_loss_gray + L1_gray

        if torch.isnan(G1_loss):
            print("NaN detected in G1_loss")
            continue
        
        
        optimizer_g1.zero_grad()
        g1_scaler.scale(G1_loss).backward()
        
        
        nn_utils.clip_grad_norm_(g1.parameters(), max_norm=1.0)

        
        g1_scaler.step(optimizer_g1)
        g1_scaler.update()
        #print(f"G1_loss version after backward: {G1_loss._version}")
        
        

        # Train Generator 2
        with torch.cuda.amp.autocast(dtype=torch.float16):
            #this step is necessary, gradient computation is tricky... 
            y_fake_gray = g1(x)
            y_fake_color = g2(y_fake_gray)
            ################################################################################
            
            D_fake_logits_color = d1(x, y_fake_color)
            D_fake_logits_color = torch.clamp(D_fake_logits_color, min=1e-7, max=1-1e-7)
            ###################################################################
            #print(f"D_fake_logits_color version before backward: {D_fake_logits_color._version}")
            G2_fake_loss_color = bce(D_fake_logits_color, torch.ones_like(D_fake_logits_color))
            L1_color = l1(y_fake_color, y) * config.L1_LAMBDA
            G2_loss = G2_fake_loss_color + L1_color

        if torch.isnan(G2_loss):
            print("NaN detected in G2_loss")
            continue
        
        optimizer_g2.zero_grad()
        g2_scaler.scale(G2_loss).backward()
        
        nn_utils.clip_grad_norm_(g2.parameters(), max_norm=1.0)
        
        g2_scaler.step(optimizer_g2)
        g2_scaler.update()
        #print(f"G2_loss version after backward: {G2_loss._version}")
        
       

    
        

    print("TRAIN \n G1_loss: ", G1_loss.item(), "\n G2_loss: ", G2_loss.item(), "\n D_loss: ", D_loss.item())
    return G1_loss.item(), G2_loss.item(), D_loss.item()
     
        
def validate_loop(d1, g1, g2, loader, l1, bce, epoch):
    g1.eval()
    g2.eval()
    d1.eval()
        
    g1_loss = 0
    g2_loss = 0
    d_loss = 0
    
    
    
    
    
        
    with torch.no_grad():
        for idx, (x, y, y_gray) in enumerate(loader):
            x, y, y_gray = x.to(config.DEVICE), y.to(config.DEVICE), y_gray.to(config.DEVICE)
                
            y_fake_gray = g1(x)
            y_fake_color = g2(y_fake_gray)
                
            D_real_logits = d1(x, y)
            D_fake_logits_color = d1(x, y_fake_color.detach())
            D_fake_logits_gray = d1(x, y_fake_gray.detach())
                
            D_real_loss = bce(D_real_logits, torch.ones_like(D_real_logits))
            D_fake_loss_color = bce(D_fake_logits_color, torch.zeros_like(D_fake_logits_color))
            D_fake_loss_gray = bce(D_fake_logits_gray, torch.zeros_like(D_fake_logits_gray))
                
            D_loss = D_real_loss + D_fake_loss_color 
                
            G_fake_loss_gray = bce(D_fake_logits_gray, torch.ones_like(D_fake_logits_gray))
            L1_gray = l1(y_fake_gray, y_gray) * config.L1_LAMBDA
            G_loss = G_fake_loss_gray + L1_gray
            
            G_fake_loss_color = bce(D_fake_logits_color, torch.ones_like(D_fake_logits_color))
            L1_color = l1(y_fake_color, y) * config.L1_LAMBDA
            G2_loss = G_fake_loss_color + L1_color
                
            g1_loss += G_loss.item()
            g2_loss += G2_loss.item()
            d_loss += D_loss.item()
        
        g1_loss /= len(loader)
        g2_loss /= len(loader)
        d_loss /= len(loader)
        
        print("VALIDATION \n G1_loss: ", g1_loss, "\n G2_loss: ", g2_loss, "\n D_loss: ", d_loss)
        
        return g1_loss, g2_loss, d_loss
        
        
        
        
        

    
     
    
    
    
    

def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3).to(config.DEVICE)
    gen2 = Generator(in_channels=3).to(config.DEVICE)
    
    
    
    #can configure different learning rate for gen and disc
    optimizer_disc = optim.Adam(disc.parameters(), lr=0.0002, betas=config.BETAS) #note betas is a play with momentum can chang here 
    optimizer_gen = optim.Adam(gen.parameters(), lr=0.0002, betas=config.BETAS)
    optimizer_gen2 = optim.Adam(gen2.parameters(), lr=0.0002, betas=config.BETAS)
    
    
    #standard GAN loss
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    #GPloss didn't work well with patchGan
    
    
    
    #load the model for hyperparam tuning
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, optimizer_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, optimizer_disc, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN2, gen2, optimizer_gen2, config.LEARNING_RATE)
        
    
    test_dataset = MapDataset_Multi(sketch_dir='sketch_test_studentSaved', target_dir='photos_test_student_whiteBG')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print("Test dataset loaded")
    
    if config.TEST_ONLY and config.LOAD_MODEL:
        
        name = "Multi_FirstTrial"
        if not os.path.exists(f"Final_Generation/g_{name}"):
            os.makedirs(f"Final_Generation/g_{name}")
        final_save_all(gen, gen2, test_loader, folderName=f"Final_Generation/g_{name}")
        
        exit()
    
    
    
    #LOAD DATASET and Save transformed images 
    train_dataset = MapDataset_Multi(sketch_dir='all_sketch_resize', target_dir='all_colorful_resize')
    ############################################################################
    #Note: may only need to run once if train_dataset didn't change
    #after applied resize and normalize, save the transformed images
    #THIS STEP ONLY FOR BACKGROUND REMOVAL!!! AND GrayScale conversion! 
    ############################################################################
    #save_transformed_images(train_dataset, save_sketch_dir='sketch_train_studentSaved', save_tar_dir='photos_train_studentSaved')
    
    
    
    #construct dataloader
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=config.NUM_WORKERS)
    
    print("Train dataset loaded")
    
    
    
    
    
    
    
    #exit()
    
    #verify val dataset
    #modify functino to save intermediate generated img 
    
    #load validation dataset
    val_dataset = MapDataset_Multi(sketch_dir='sketch_val_studentSaved', target_dir='photos_val_student_whiteBG')
    ############################################################################
    #save_transformed_images(val_dataset, save_sketch_dir='sketch_val_studentSaved', save_tar_dir='photos_val_studentSaved')
    ############################################################################
    
    
    
   
    
    
    
    
    
    #only validate one img at a time
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)

    print("Val dataset loaded")
    
     # Visualize a batch of dataset pairs in train_loader
    sample_batch = next(iter(val_loader))
    x, y, y_gray = sample_batch[0], sample_batch[1], sample_batch[2]
    print(x.shape, y.shape, y_gray.shape)
    print(x.dtype, y.dtype, y_gray.dtype)
    
    '''
    #set batchsize to corresponded batch size of val/train loader
    fig, axes = plt.subplots(4, 3, figsize=(10, 10))
    for i in range(4):
        axes[i, 0].imshow(x[i].permute(1, 2, 0))
        axes[i, 0].set_title('Sketch')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(y[i].permute(1, 2, 0))
        axes[i, 1].set_title('Target Image')
        axes[i, 1].axis('off')
        axes[i, 2].imshow(y_gray[i].permute(1, 2, 0))
        axes[i, 2].set_title('Gray Image')
        axes[i, 2].axis('off')
    plt.tight_layout()
    plt.show()
    '''
    
    
    
    #perform float16 training
    g1_scaler = torch.cuda.amp.GradScaler()
    g2_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    
    #save all loss for plotting
    G1_train_loss_all = np.zeros(config.NUM_EPOCHS)
    G2_train_loss_all = np.zeros(config.NUM_EPOCHS)
    D_train_loss_all = np.zeros(config.NUM_EPOCHS)
    G1_Val_loss_all = np.zeros(config.NUM_EPOCHS)
    G2_Val_loss_all = np.zeros(config.NUM_EPOCHS)
    D_Val_loss_all = np.zeros(config.NUM_EPOCHS)
    
    
    start_time = time.time()
    
    #saving name
    #####################################
    name = config.NAME
    #####################################
    
    #train the model
    for epoch in tqdm(range(config.NUM_EPOCHS)):
        
        
        print("\n Epoch ", epoch)
        G1_loss, G2_loss, D_loss = train_loop(disc, gen, gen2,  train_loader, optimizer_disc, optimizer_gen, optimizer_gen2, L1_LOSS, BCE, g1_scaler, g2_scaler, d_scaler)
        G1_val_loss, G2_val_loss, D_val_loss = validate_loop(disc, gen, gen2, val_loader, L1_LOSS, BCE, epoch)
        
        
        
        
        
        
        
        
        G1_train_loss_all[epoch] = G1_loss
        G2_train_loss_all[epoch] = G2_loss
        D_train_loss_all[epoch] = D_loss
        G1_Val_loss_all[epoch] = G1_val_loss
        G2_Val_loss_all[epoch] = G2_val_loss
        D_Val_loss_all[epoch] = D_val_loss
        
        
        #save model checkpoint every # epoch
        #and save the last model config 
        if (config.SAVE_MODEL and epoch % 30 == 0 and epoch != 0) or (config.SAVE_MODEL and epoch == config.NUM_EPOCHS - 1):
            save_checkpoint(gen, optimizer_gen, filename=f"Generators/g1_{name}_epoch_{epoch}.pth.tar")
            save_checkpoint(disc, optimizer_disc, filename=f"Discriminators/d_{name}_epoch_{epoch}.pth.tar")
            save_checkpoint(gen2, optimizer_gen2, filename=f"Generators/g2_{name}_epoch_{epoch}.pth.tar")
            
        
        #save some validation generated examples
        if epoch % 10 == 0 or epoch == config.NUM_EPOCHS - 1 or epoch == 0:
            # Create directory if it doesn't exist
            folder = f"validation_generated_examples_{name}"
            if not os.path.exists(folder):
                os.makedirs(folder)

            save_gray_and_color_examples(gen, gen2, val_loader, epoch, folder)
            
            
            if epoch == config.NUM_EPOCHS - 1:
                if not os.path.exists(f"Final_Generation/g_{name}"):
                    os.makedirs(f"Final_Generation/g_{name}")
                final_save_all(gen, gen2, test_loader, folderName=f"Final_Generation/g_{name}")
    
    end_time = time.time()
    print(f"Time taken to Train: {end_time - start_time}")
    
    np.savetxt(f"History/G1_train_loss_{name}.csv", G1_train_loss_all)
    np.savetxt(f"History/G2_train_loss_{name}.csv", G2_train_loss_all)
    np.savetxt(f"History/D_train_loss_{name}.csv", D_train_loss_all)
    np.savetxt(f"History/G1_Val_loss_{name}.csv", G1_Val_loss_all)
    np.savetxt(f"History/G2_Val_loss_{name}.csv", G2_Val_loss_all)
    np.savetxt(f"History/D_Val_loss_{name}.csv", D_Val_loss_all)
    
    print("All history saved")
    
    
if __name__ == "__main__":
    main()