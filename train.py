#from lib
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


#from files
import config
from utils import save_checkpoint, load_checkpoint, save_some_examples
from dataset import MapDataset, save_transformed_images, save_single_transformed
from generator import Generator
from discriminator import Discriminator


def train_loop(disc, gen, loader, optimizer_disc, optimizer_gen, l1, bce, g_scaler,d_scaler):
    
    
    #(x,y) is a batch of sketch (x) and target img (y)
    #idx represent current batch #
    for idx, (x,y) in enumerate(loader):
        
        #print max pixel value of x and y
        #print(x.max(), y.max(), x.min(), y.min())
        #exit()
        
        #x = x / 255.0
        #y = y/255.0
         
        
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        
        '''
        #debug 
        print(x.type(), y.type())
        x = x.to("cpu")
        y = y.to("cpu")
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(x[0].permute(1, 2, 0))
        axes[0].set_title('Sketch')
        axes[0].axis('off')
        axes[1].imshow(y[0].permute(1, 2, 0))
        axes[1].set_title('Target Image')
        axes[1].axis('off')
        plt.show()
        exit()
        '''
        
        
        
    
    
        
        #print(x.type(), y.type())
        
        #print("Batch: ", idx)
        
        #train discriminator
        #cast operation to mixed precision float16
        with torch.cuda.amp.autocast(dtype=torch.float16):
            y_fake = gen(x)
            #get the probability of real img and fake img
            D_real_logits = disc(x,y)
            D_fake_logits = disc(x,y_fake.detach()) #detach is necessary 
            #evalulate loss, since pass in all real imgs, the target is all 1 (use torch.ones_like to create tensor of 1s with same shape as D_real_logits)
            D_real_loss = bce(D_real_logits, torch.ones_like(D_real_logits))
            #same for fake loss, except target all 0
            D_fake_loss = bce(D_fake_logits, torch.zeros_like(D_fake_logits))
            
            
            #sometimes ppl divide the loss by 2....................
            D_loss = (D_real_loss + D_fake_loss)
        
        disc.zero_grad()
        #steps to perform float16 training
        d_scaler.scale(D_loss).backward()
        d_scaler.step(optimizer_disc)
        d_scaler.update()
        
        
        
        #train generator
        with torch.cuda.amp.autocast(dtype=torch.float16):
            #trying to fool discriminator, therefore the fake_logit computed from disc should have target all of 1 !!!!
            #Note while in disc training, want to align fake_logit with 0!!!! Battle... 
            D_fake_logits = disc(x,y_fake)
            G_fake_loss = bce(D_fake_logits, torch.ones_like(D_fake_logits))
            
            #sum of diff in y_pred and y
            L1 = l1(y_fake, y) * config.L1_LAMBDA
            #total loss
            G_loss = G_fake_loss + L1
        
        
        optimizer_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(optimizer_gen)
        g_scaler.update()
        
        
    print("G_loss: ", G_loss.item(), "D_loss: ", D_loss.item())
        
        
    
    
     
    
    
    
    

def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3).to(config.DEVICE)
    
    
    
    #can configure different learning rate for gen and disc
    optimizer_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS) #note betas is a play with momentum can chang here 
    optimizer_gen = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS)
    
    #standard GAN loss
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    #GPloss didn't work well with patchGan
    
    
    
    #load the model for hyperparam tuning
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, optimizer_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, optimizer_disc, config.LEARNING_RATE)
    
    
    
    #LOAD DATASET and Save transformed images 
    train_dataset = MapDataset(sketch_dir='sketches_train_student', target_dir='photos_train_student')
    ############################################################################
    #Note: may only need to run once if train_dataset didn't change
    #after applied resize and normalize, save the transformed images
    #THIS STEP ONLY FOR BACKGROUND REMOVAL!!! AND GrayScale conversion! 
    ############################################################################
    
    
    #save_transformed_images(train_dataset, save_sketch_dir='train/transformed_sketch', save_tar_dir='train/transformed_photo')
    #construct dataloader
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    
    print("Train dataset loaded")
    
    
    
    '''
    # Visualize random pair of images in train_loader
    sample_batch = next(iter(train_loader))
    x, y = sample_batch[0][0], sample_batch[1][0]

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(x.permute(1, 2, 0))
    axes[0].set_title('Sketch')
    axes[0].axis('off')
    axes[1].imshow(y.permute(1, 2, 0))
    axes[1].set_title('Target Image')
    axes[1].axis('off')
    plt.show()
    exit()
    '''
    
    
    #load validation dataset
    val_dataset = MapDataset(sketch_dir='sketches_val_student', target_dir='photos_val_student')
    ############################################################################
    #save_transformed_images(val_dataset, save_sketch_dir='val/transformed_sketch', save_tar_dir='val/transformed_photo')
    ############################################################################
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print("Val dataset loaded")
    
    
    #perform float16 training
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    
    #train the model
    for epoch in tqdm(range(config.NUM_EPOCHS)):
        
        print("Epoch: ", epoch)
        
        train_loop(disc, gen, train_loader, optimizer_disc, optimizer_gen, BCE, L1_LOSS, g_scaler, d_scaler)
        
        #save model checkpoint every 5 epoch
        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, optimizer_gen, filename=f"gen_epoch_{epoch}.pth.tar")
            save_checkpoint(disc, optimizer_disc, filename=f"disc_epoch_{epoch}.pth.tar")
            
        
        #save some validation generated examples
        save_some_examples(gen, val_loader, epoch, folder="validation_generated_examples2")
    
if __name__ == "__main__":
    main()