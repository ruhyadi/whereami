"""
Train script
"""
import os
from tqdm import tqdm
import argparse
from comet_ml import Experiment
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from torch import nn

from dataset import WhereIamDataset
from model import Model

def train(
    dataset_path='data',
    weights_path='weights',
    frame_per_minute=10,
    backbone='resnet18', 
    val_size=0.2, 
    epochs=10,
    val_epoch=5, 
    batch_size=32, 
    num_workers=4,
    lr=0.001,
    save_epoch=5,
    api_key='xxx'):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAutocontrast(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    params_loader = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': num_workers
    }

    # comet ml experiment
    experiment = Experiment(api_key, project_name="WhereAmI")
    experiment.log_parameters(params_loader)

    dataset = WhereIamDataset(dataset_path, frame_per_minute, transform)

    train_set = int((1-val_size) * len(dataset))
    val_set = int(len(dataset) - train_set)
    trainset, valset = random_split(dataset, [train_set, val_set])

    train_loader = DataLoader(trainset, **params_loader)
    val_loader = DataLoader(valset, **params_loader)
    data_loader = {'train': train_loader, 'val': val_loader}
    
    # model
    class_list = dataset.class_list()
    model = Model(backbone, class_list).to(device)

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # loss function
    criterion = nn.MSELoss().to(device)

    # load previous weights
    latest_model = None
    first_epoch = 1
    if not os.path.isdir(weights_path):
        os.makedirs(weights_path)
    else:
        try:
            latest_model = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')][-1]
        except:
            pass
    
    if latest_model is not None:
        checkpoint = torch.load(os.path.join(weights_path, latest_model))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        first_epoch = checkpoint['epoch']
        criterion = checkpoint['loss']

        print(f'[INFO] Using previous model {latest_model} at {first_epoch} epochs')
        print('[INFO] Resuming training...')

    with experiment.train():
        step = 1
        for epoch in range(first_epoch, epochs+1):
            # model mode
            phase = 'train'
            model.train()
            if epoch % val_epoch == 0:
                phase = 'val'
                model.eval()

            # loop thru batch
            with tqdm(data_loader[phase], unit='batch', desc=f'{phase} | Epoch {epoch}') as tepoch:
                for local_batch, local_label in tepoch:
                    local_batch = local_batch.float().to(device)
                    local_label = local_label.float().to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # prediction
                    y_pred = model(local_batch)

                    # compute loss
                    loss = criterion(y_pred, local_label)

                    # backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        writer.add_scalar('train_loss/step', loss, step)
                        writer.add_scalar('train_loss/epoch', loss, epoch)
                        experiment.log_metric('train_loss', loss, step=step, epoch=epoch)
                    else:
                        writer.add_scalar('val_loss/step', loss, step)
                        writer.add_scalar('val_loss/epoch', loss, epoch)
                        experiment.log_metric('val_loss', loss, step=step, epoch=epoch)

                    # update progress bar and comet ml
                    tepoch.set_postfix(loss=loss.item())
                    step = step + 1

            if epoch % save_epoch == 0:
                model_name = os.path.join(weights_path, f'{backbone}_epoch_{epoch}.pkl')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                }, model_name)
                print(f'[INFO] Saving weights as {model_name}')
    
    writer.flush()
    writer.close()

def main():
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('--dataset_path', type=str, default='data', help='dataset path')
    parser.add_argument('--weights_path', type=str, default='weights', help='weights path')
    parser.add_argument('--frame_per_minute', type=int, default=10, help='frame per minute')
    parser.add_argument('--backbone', type=str, default='resnet18', help='model backbone')
    parser.add_argument('--val_size', type=float, default=0.2, help='validation size')
    parser.add_argument('--epochs', type=int, default=10, help='num of epochs')
    parser.add_argument('--val_epoch', type=int, default=5, help='doing validation every n epoch')
    parser.add_argument('--batch_size', type=int, default=32, help='size of batch')
    parser.add_argument('--num_workers', type=int, default=2, help='number of CPU')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--save_epoch', type=int, default=5, help='save model every n epoch')
    parser.add_argument('--api_key', type=str, default='xxx', help='Comet ML API key')
    args = parser.parse_args()

    train(**vars(args))

if __name__ == "__main__":
    main()