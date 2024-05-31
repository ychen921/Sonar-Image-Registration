import torch
import torch.nn
from torchsummary import summary
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import argparse
import time
from tqdm import tqdm

from Mics.utils import plot_loss
from Mics.metrics import NCC, dice_score
from Networks.AIRNet import AIRNet
from Networks.InceptionNet import InceptionNet
from Networks.DenseNet import DenseNet
from DataReader.DataReader import SonarPairDataset

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

loss_func = NCC()


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--NumEpochs', type=int, default=10, 
                        help='Number of Epochs to Train for, Default:10')
    
    Parser.add_argument('--CkptsPath', dest='CkptsPath', default='/home/ychen921/808E/final_project/Dense_ckpts', 
                        help='Path to load latest model from, Default:/home/ychen921/808E/final_project/Inception_ckpts')
    
    Parser.add_argument('--BasePath', dest='BasePath', default='/home/ychen921/808E/final_project/Dataset/Test',
                        help='Path to load images from, Default:/home/ychen921/808E/final_project/Dataset/Test')
    
    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    CkptsPath = Args.CkptsPath
    BasePath = Args.BasePath

    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to 128x128
        transforms.ToTensor()            # Convert images to tensors
    ])

    SonarPair = SonarPairDataset(data_folder=BasePath, transform=transform)
    data_loader = DataLoader(SonarPair, batch_size=32, shuffle=False)

    # model = AIRNet().to(device)
    # model = InceptionNet().to(device)
    model = DenseNet().to(device)

    loss_values = []
    dice_values = []

    for epoch in range(NumEpochs):
        saved_model = CkptsPath + '/' + str(epoch) + '_model.pt'

        # Load checkpoint
        checkpoint = torch.load(saved_model)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Evaluation mode
        model.eval()

        Epoch_loss = []
        Epoch_dice = []

        with torch.no_grad():
            for i, (fix_img, mov_img) in enumerate(tqdm(data_loader)):
                fix_img = (fix_img/255.0).to(device)
                mov_img = (mov_img/255.0).to(device)

                wraped, _ = model(fix_img, mov_img)

                loss = loss_func(fix_img, wraped)
                dice = dice_score(wraped, fix_img)
                Epoch_loss.append(loss.item())
                Epoch_dice.append(dice.item())

        LossThisEpoch = sum(Epoch_loss) / len(Epoch_loss)
        loss_values.append(LossThisEpoch)

        DiceThisEpoch = sum(Epoch_dice) / len(Epoch_dice)
        dice_values.append(DiceThisEpoch*1e3)

        print('Epoch:{}, NCC Loss:{}, Dice:{}'.format(epoch+1, LossThisEpoch, DiceThisEpoch*1e3))

    plot_loss(loss_values, dice_values)

if __name__ == '__main__':
    main()