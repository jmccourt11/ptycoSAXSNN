import os
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import colors
import sys
from tqdm import tqdm
import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt
import sys
import os
import importlib

plt.rcParams["image.cmap"] = "jet"


# Setting path
# path = Path("Y:/ptychosaxs")  # /net/micdata/data2/12IDC mounted windows drive
path = Path("/net/micdata/data2/12IDC/ptychosaxs/")
#path = Path("/scratch/")
# Join paths
MODEL_SAVE_PATH = path / 'batch_mode_250/trained_model/' # Automatically adds the correct separator
if (not os.path.isdir(MODEL_SAVE_PATH)):
    os.mkdir(MODEL_SAVE_PATH)
print(MODEL_SAVE_PATH)


lattice_list=['ClathII','SC',]
noise_list=['Noise','noNoise']
unet_status_list=['no_Unet','Unet']
loss_function_list=['pearson_loss','L2','L1']
probe_size_list=[256]#[128,256]

numDPs_list=[9600]
symmetry_weight=0.0



# For full training
EPOCHS = 250
# Specify which GPUs to use (e.g. GPUs 1 and 2)
selected_gpus = [0,1]  # Select GPUs (0, 1, 2, or 3)
NGPUS = len(selected_gpus)
#NGPUS = torch.cuda.device_count() # if all GPUs are used
BATCH_SIZE = NGPUS*16
LR = NGPUS * 1e-3
print("GPUs:", NGPUS, "Batch size:", BATCH_SIZE, "Learning rate:", LR)
no_probe=True

# Add the models directory to the path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../src/models/')))



def pearson_loss(output, target):
    """
    Compute 1 - Pearson correlation coefficient as a loss function, with additional
    penalty for breaking inversion symmetry outside central 64x64 circle.
    Args:
        output: Predicted values (B, C, H, W)
        target: Target values (B, C, H, W)
    Returns:
        loss: Combined loss of correlation and symmetry terms
    """
    # Create circular mask for central 64x64 region
    h, w = output.shape[2:]
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
    center_y, center_x = h // 2, w // 2
    radius = 32  # 64/2 for 64x64 circle
    mask = ((y - center_y)**2 + (x - center_x)**2 <= radius**2).to(output.device)
    
    # Basic Pearson correlation loss
    output_flat = output.view(output.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    output_mean = output_flat.mean(dim=1, keepdim=True)
    target_mean = target_flat.mean(dim=1, keepdim=True)
    
    output_centered = output_flat - output_mean
    target_centered = target_flat - target_mean
    
    numerator = (output_centered * target_centered).sum(dim=1)
    denominator = torch.sqrt((output_centered**2).sum(dim=1) * (target_centered**2).sum(dim=1))
    correlation = numerator / (denominator + 1e-8)
    pearson_loss = 1 - correlation.mean()
    
    # Symmetry loss for region outside circle
    output_flipped = torch.flip(output, dims=[-2, -1])  # Flip both height and width
    outside_mask = ~mask
    symmetry_loss = torch.mean(torch.abs(output[..., outside_mask] - output_flipped[..., outside_mask]))
    
    # Combine losses - can adjust weight of symmetry term
    #symmetry_weight = 0.5
    total_loss = pearson_loss + symmetry_weight * symmetry_loss
    
    return total_loss, pearson_loss, symmetry_loss



for probe_size in probe_size_list:
    for lattice in lattice_list:
        for noise in noise_list:
            directory=f'Lattice{lattice}_Probe{probe_size}x{probe_size}_ZCB_9_3D__{noise}_sim_ZCB_9_3D_S5065_N600_steps4_dp256'
            print(directory)
            # Load the data
            #data_path = os.path.abspath(os.path.join(os.getcwd(), f'/scratch/preprocessed_sim_{directory}.npz'))
            data_path = f'/net/micdata/data2/12IDC/ptychosaxs/batch_mode/preprocessed/preprocessed_sim_{directory}.npz'
            print('Loading data from:', data_path)
            data = np.load(data_path)

            # Extract the arrays
            amp_conv_red = data['amp_conv_red']
            amp_ideal_red = data['amp_ideal_red']
            amp_probe_red = data['amp_probe_red']
            
            # Set the number of patterns in test, train or validation set
            NTEST = amp_conv_red.shape[0]//4
            NTRAIN = amp_conv_red.shape[0]-NTEST
            NVALID = NTEST//2 # NTRAIN//

            print(NTRAIN,NTEST,NVALID)

            H,W=amp_ideal_red[0].shape[0],amp_ideal_red[0].shape[1]
            print(H,W)
            
            tst_start = amp_conv_red.shape[0]-NTEST

            #separate data and convert to tensors and shuf
            X_train = amp_conv_red[:NTRAIN].reshape(-1,H,W)[:,np.newaxis,:,:]
            X_test = amp_conv_red[tst_start:].reshape(-1,H,W)[:,np.newaxis,:,:]

            Xp_train = amp_probe_red[:NTRAIN].reshape(-1,H,W)[:,np.newaxis,:,:]
            Xp_test = amp_probe_red[tst_start:].reshape(-1,H,W)[:,np.newaxis,:,:]

            Y_I_train = amp_ideal_red[:NTRAIN].reshape(-1,H,W)[:,np.newaxis,:,:]
            Y_I_test = amp_ideal_red[tst_start:].reshape(-1,H,W)[:,np.newaxis,:,:]

            ntrain=X_train.shape[0]
            ntest=X_test.shape[0]

            X_train, Xp_train, Y_I_train = shuffle(X_train, Xp_train, Y_I_train, random_state=0)

            #Training data
            X_train_tensor = torch.Tensor(X_train)
            Xp_train_tensor = torch.Tensor(Xp_train) 
            Y_I_train_tensor = torch.Tensor(Y_I_train) 

            #Test data
            X_test_tensor = torch.Tensor(X_test)
            Xp_test_tensor = torch.Tensor(Xp_test) 
            Y_I_test_tensor = torch.Tensor(Y_I_test) 

            print(X_train_tensor.shape,Xp_train_tensor.shape, Y_I_train_tensor.shape)

            if no_probe:
                train_data = TensorDataset(X_train_tensor,Y_I_train_tensor)
                test_data = TensorDataset(X_test_tensor,Xp_test_tensor)
            else:
                train_data = TensorDataset(X_train_tensor,Xp_train_tensor,Y_I_train_tensor)
                test_data = TensorDataset(X_test_tensor,Xp_test_tensor)

            N_TRAIN = X_train_tensor.shape[0]

            train_data2, valid_data = torch.utils.data.random_split(train_data,[N_TRAIN-NVALID,NVALID])
            print(len(train_data2),len(train_data2[0]),len(valid_data),len(test_data))

            #download and load training data
            trainloader = DataLoader(train_data2, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

            validloader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

            #same for test
            #download and load training data
            testloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

            
            
            for unet_status in unet_status_list:
                    
                for loss_function in loss_function_list:
                    

                    # First, try to import the module
                    if unet_status=='no_Unet':
                        try:
                            import encoder1_no_Unet
                            # Force reload the module
                            importlib.reload(encoder1_no_Unet)
                            # Now import the class from the freshly reloaded module
                            from encoder1_no_Unet import recon_model
                            print("Successfully imported recon_model_no_Unet")
                        except Exception as e:
                            print(f"Import error: {e}")
                    else:
                        #First, try to import the module
                        try:
                            import encoder1
                            importlib.reload(encoder1)
                            # Now import the class from the freshly reloaded module
                            from encoder1 import recon_model
                            print("Successfully imported recon_model_Unet")
                        except Exception as e:
                            print(f"Import error: {e}")


                    model = recon_model()

                    
                    if no_probe:
                        for ampsI,ampsO in trainloader:
                            print("batch size:", ampsI.shape)
                            amp = model(ampsI)#,ampsP)
                            print(amp.shape)
                            print(amp.dtype)
                            break
                    else:
                        for ampsI,ampsP,ampsO in trainloader:
                            print("batch size:", ampsI.shape)
                            amp = model(ampsI,ampsP)
                            print(amp.shape)
                            print(amp.dtype)
                            break    

                    device = torch.device(f"cuda:{selected_gpus[0]}" if torch.cuda.is_available() else "cpu")
                    device_ids = selected_gpus  # This will use only the specified GPUs

                    if NGPUS > 1:
                        print("Let's use", NGPUS, "GPUs!")
                        model = nn.DataParallel(model, device_ids=device_ids)  # Explicitly specify which GPUs to use

                    model = model.to(device)
                    print(model)        
            
                    print(f'{directory}_{unet_status}_{loss_function}')
                    
                    #Optimizer details
                    iterations_per_epoch = np.floor((NTRAIN-NVALID)/BATCH_SIZE)+1 #Final batch will be less than batch size
                    step_size = 6*iterations_per_epoch 
                    print(iterations_per_epoch)
                    print("LR step size is:", step_size, "which is every %d epochs" %(step_size/iterations_per_epoch))



                    if loss_function=='L1':
                        criterion = nn.L1Loss()
                    elif loss_function=='L2':
                        criterion = nn.MSELoss()
                    else:
                        print('Using pearson_loss')
                        

                    optimizer = torch.optim.Adam(model.parameters(), lr = LR, weight_decay=1e-5)
                    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=LR/10, max_lr=LR, step_size_up=step_size,
                                                                cycle_momentum=False, mode='triangular2')
                                                                
                                                                
                    #Function to update saved model if validation loss is minimum
                    print('Model string path')
                    print(f'{directory}_{unet_status}_{loss_function}_symmetry_{symmetry_weight}.pth')
                    def update_saved_model(model, path, current_epoch, best_val_loss):
                        if not os.path.isdir(path):
                            os.mkdir(path)
                        
                        # Save the best overall model
                        if (NGPUS>1):    
                            torch.save(model.module.state_dict(), path / f'best_model_{directory}_{unet_status}_{loss_function}_symmetry_{symmetry_weight}.pth')
                        else:
                            torch.save(model.state_dict(), path / f'best_model_{directory}_{unet_status}_{loss_function}_symmetry_{symmetry_weight}.pth')
                        
                        # Define epoch intervals (50, 100, 150, etc.)
                        epoch_intervals = [2, 10, 25, 50, 100, 150, 200, 250]#, 300, 400, 500]
                        
                        # For each interval, save the best model within that interval
                        for interval in epoch_intervals:
                            if current_epoch <= interval:
                                # Create a filename that includes the epoch interval
                                interval_filename = f'best_model_{directory}_{unet_status}_epoch_{interval}_{loss_function}_symmetry_{symmetry_weight}.pth'
                                interval_path = path / interval_filename
                                
                                # If this is the first time we're saving for this interval, save the model
                                if not interval_path.exists():
                                    if (NGPUS>1):
                                        torch.save(model.module.state_dict(), interval_path)
                                    else:
                                        torch.save(model.state_dict(), interval_path)
                                    print(f"Saving best model for epoch interval {interval} at epoch {current_epoch}")
                                # If we already have a model for this interval, only update if the current loss is better
                                else:
                                    # Load the previous best loss for this interval
                                    prev_loss_path = path / f'best_loss_epoch_{directory}_{unet_status}_{interval}_{loss_function}_symmetry_{symmetry_weight}.txt'
                                    if prev_loss_path.exists():
                                        with open(prev_loss_path, 'r') as f:
                                            prev_best_loss = float(f.read().strip())
                                        
                                        # Update if current loss is better
                                        if best_val_loss < prev_best_loss:
                                            if (NGPUS>1):
                                                torch.save(model.module.state_dict(), interval_path)
                                            else:
                                                torch.save(model.state_dict(), interval_path)
                                            with open(prev_loss_path, 'w') as f:
                                                f.write(str(best_val_loss))
                                            print(f"Updating best model for epoch interval {interval} at epoch {current_epoch}")
                                    else:
                                        # First time saving for this interval
                                        if (NGPUS>1):
                                            torch.save(model.module.state_dict(), interval_path)
                                        else:
                                            torch.save(model.state_dict(), interval_path)
                                        with open(prev_loss_path, 'w') as f:
                                            f.write(str(best_val_loss))
                                        print(f"Saving best model for epoch interval {interval} at epoch {current_epoch}")



                    def train(trainloader,metrics):
                        tot_loss = 0.0
                        tot_loss_amp = 0.0
                        tot_loss_symmetry = 0.0
                        
                        for i, (ft_images,amps) in tqdm(enumerate(trainloader)):
                            ft_images = ft_images.to(device) #Move everything to device
                            amps = amps.to(device)
                            pred_amps = model(ft_images) #Forward pass
                            
                            #Compute losses
                            #loss_a = criterion(pred_amps,amps) #Monitor amplitude loss
                            if loss_function=='pearson_loss':
                                loss_a,loss_pearson,loss_symmetry = pearson_loss(pred_amps,amps)
                            else:
                                loss_a = criterion(pred_amps,amps)
                            loss = loss_a #Use equiweighted amps and phase

                            #Zero current grads and do backprop
                            optimizer.zero_grad() 
                            loss.backward()
                            optimizer.step()
                            
                            if loss_function=='pearson_loss':
                                tot_loss += loss.detach().item()
                                tot_loss_amp += loss_pearson.detach().item()
                                tot_loss_symmetry += loss_symmetry.detach().item()
                            else:
                                tot_loss += loss.detach().item()

                            #Update the LR according to the schedule -- CyclicLR updates each batch
                            scheduler.step() 
                            metrics['lrs'].append(scheduler.get_last_lr())
                            
                            
                        #Divide cumulative loss by number of batches-- sli inaccurate because last batch is different size
                        if loss_function=='pearson_loss':
                            metrics['losses'].append([tot_loss/i,tot_loss_amp/i,tot_loss_symmetry/i]) 
                        else:
                            metrics['losses'].append([tot_loss/i]) 
                        
                    def validate(validloader,metrics):
                        tot_val_loss = 0.0
                        tot_val_loss_amp = 0.0
                        tot_val_loss_symmetry = 0.0
                        for j, (ft_images,amps) in enumerate(validloader):
                            ft_images = ft_images.to(device)
                            amps = amps.to(device)
                            pred_amps = model(ft_images) #Forward pass
                            
                            if loss_function=='pearson_loss':   
                                val_loss_a,val_loss_pearson,val_loss_symmetry = pearson_loss(pred_amps,amps)
                            else:
                                val_loss_a = criterion(pred_amps,amps)
                            
                            val_loss = val_loss_a
                        
                            if loss_function=='pearson_loss':
                                tot_val_loss += val_loss.detach().item()
                                tot_val_loss_amp += val_loss_pearson.detach().item()
                                tot_val_loss_symmetry += val_loss_symmetry.detach().item()
                            else:
                                tot_val_loss += val_loss.detach().item()
                        
                        if loss_function=='pearson_loss':
                            metrics['val_losses'].append([tot_val_loss/j,tot_val_loss_amp/j,tot_val_loss_symmetry/j])
                        else:
                            metrics['val_losses'].append([tot_val_loss/j])
                    
                        #Update saved model if val loss is lower
                        if(tot_val_loss/j<metrics['best_val_loss']):
                            print("Saving improved model after Val Loss improved from %.5f to %.5f" %(metrics['best_val_loss'],tot_val_loss/j))
                            metrics['best_val_loss'] = tot_val_loss/j
                            update_saved_model(model, MODEL_SAVE_PATH, metrics['current_epoch'], tot_val_loss/j)
                        
                    # Initialize metrics dictionary with current_epoch
                    metrics = {
                        'losses': [],
                        'val_losses': [],
                        'lrs': [],
                        'best_val_loss': float('inf'),
                        'current_epoch': 0
                    }

                    for epoch in range(EPOCHS):
                        metrics['current_epoch'] = epoch  # Update current epoch in metrics
                        
                        #Set model to train mode
                        model.train() 
                        #Training loop
                        train(trainloader,metrics)
                        
                        #Switch model to eval mode
                        model.eval()
                        
                        #Validation loop
                        validate(validloader,metrics)
                        if loss_function=='pearson_loss':
                            print('Epoch: %d | Total  | Train Loss: %.5f | Val Loss: %.5f' %(epoch, metrics['losses'][-1][0], metrics['val_losses'][-1][0]))
                            print('Epoch: %d | Amp | Train Loss: %.5f | Val Loss: %.5f' %(epoch, metrics['losses'][-1][1], metrics['val_losses'][-1][1]))
                            print('Epoch: %d | Symmetry | Train Loss: %.5f | Val Loss: %.5f' %(epoch, metrics['losses'][-1][2], metrics['val_losses'][-1][2]))
                            print('Epoch: %d | Ending LR: %.6f ' %(epoch, metrics['lrs'][-1][0]))
                        else:
                            print('Epoch: %d | Total  | Train Loss: %.5f | Val Loss: %.5f' %(epoch, metrics['losses'][-1][0], metrics['val_losses'][-1][0]))
                            print('Epoch: %d | Ending LR: %.6f ' %(epoch, metrics['lrs'][-1][0]))
                            
                    batches = np.linspace(0,len(metrics['lrs']),len(metrics['lrs'])+1)
                    epoch_list = batches/iterations_per_epoch

                    plt.plot(epoch_list[1:],metrics['lrs'], 'C3-')
                    plt.grid()
                    plt.ylabel("Learning rate")
                    plt.xlabel("Epoch")

                    losses_arr = np.array(metrics['losses'])
                    val_losses_arr = np.array(metrics['val_losses'])
                    losses_arr.shape
                    fig, ax = plt.subplots(1,sharex=True, figsize=(15, 8))
                    ax.plot(losses_arr[:,0], 'C3o-', label = "Total Train loss")
                    ax.plot(val_losses_arr[:,0], 'C0o-', label = "Total Val loss")
                    ax.set(ylabel='Loss')
                    ax.grid()
                    ax.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
                    plt.tight_layout()
                    plt.xlabel("Epochs")
                    #plt.savefig(f'/scratch/plots/{directory}_{unet_status}_{loss_function}_symmetry_{symmetry_weight}_train_loss.png', bbox_inches='tight', dpi=300)
                    plt.savefig(f'/net/micdata/data2/12IDC/ptychosaxs/batch_mode_250/plots/{directory}_{unet_status}_{loss_function}_symmetry_{symmetry_weight}_train_loss.png', bbox_inches='tight', dpi=300)
                    plt.close()
                    #plt.show()
                                            

                    model.eval()
                    results = []
                    for i, test in enumerate(testloader):
                        tests = test[0].to(device)
                        testsp = test[1].to(device)
                        result = model(tests)
                        for j in range(tests.shape[0]):
                            results.append(result[j].detach().to("cpu").numpy())
                            
                    results = np.array(results).squeeze()

                    h,w = H,W
                    ntest=results.shape[0]
                    plt.figure()
                    n = 5
                    f,ax=plt.subplots(4,n,figsize=(15, 12))
                    plt.gcf().text(0.02, 0.8, "Input", fontsize=20)
                    plt.gcf().text(0.02, 0.6, "True I", fontsize=20)
                    plt.gcf().text(0.02, 0.4, "Predicted I", fontsize=20)
                    plt.gcf().text(0.02, 0.2, "Difference I", fontsize=20)

                    for i in range(0,n):
                        j=int(round(np.random.rand()*(ntest-1)))

                        # display FT
                        im=ax[0,i].imshow(X_test[j].reshape(h, w), vmin=0, vmax=1)
                        plt.colorbar(im, ax=ax[0,i], format='%.2f')
                        ax[0,i].get_xaxis().set_visible(False)
                        ax[0,i].get_yaxis().set_visible(False)

                        # display original intens
                        im=ax[1,i].imshow(Y_I_test[j].reshape(h, w))#, vmin=0, vmax=1)
                        plt.colorbar(im, ax=ax[1,i], format='%.2f')
                        ax[1,i].get_xaxis().set_visible(False)
                        ax[1,i].get_yaxis().set_visible(False)
                        
                        # display predicted intens
                        im=ax[2,i].imshow(results[j].reshape(h, w))#, vmin=0.0, vmax=1)
                        plt.colorbar(im, ax=ax[2,i], format='%.2f')
                        ax[2,i].get_xaxis().set_visible(False)
                        ax[2,i].get_yaxis().set_visible(False)

                        #Difference in amplitude
                        im=ax[3,i].imshow(Y_I_test[j].reshape(h, w)-results[j].reshape(h, w), 
                                        vmin=-0.5, vmax=0.5, cmap='RdBu')
                        plt.colorbar(im, ax=ax[3,i], format='%.2f')
                        ax[3,i].get_xaxis().set_visible(False)
                        ax[3,i].get_yaxis().set_visible(False)
                    #plt.savefig(f'/scratch/plots/{directory}_{unet_status}_{loss_function}_symmetry_{symmetry_weight}_test.png', bbox_inches='tight', dpi=300)
                    plt.savefig(f'/net/micdata/data2/12IDC/ptychosaxs/batch_mode_250/plots/{directory}_{unet_status}_{loss_function}_symmetry_{symmetry_weight}_test.png', bbox_inches='tight', dpi=300)
                    plt.close()
                    #plt.show()             