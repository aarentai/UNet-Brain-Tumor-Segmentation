import torch
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import BraTS_dataset
from model import UNet
from loss import DiceLoss
import csv

num_epochs = 100
num_workers = 4 # read data by multithread
batch_size = 2  


def train(model):
    # continue or not
    resume = True

    # record dice loss of training process
    csvFile = open("training_process_0111_in.csv", "w")
    writer = csv.writer(csvFile)
    writer.writerow(['epoch', 'batch', 'dice_loss'])

    model.train()
    dataset = BraTS_dataset('data/', 'images/', 'labels/', transform=None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # criterion = nn.CrossEntropyLoss()
    criterion = DiceLoss()
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=0.00001) #1e-4 0.00001

    if resume:
        checkpoint = torch.load('/home/sci/hdai/Projects/UNet/epoch_693_checkpoint.pth.tar')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # training process
    for epoch in range(1, num_epochs + 1):
        # loader as iterator
        f = open("./0111-701-800.txt", 'a')
        f.write(str(epoch+700)+'\n')
        print(epoch+700)
        for i_batch, sample_batched in enumerate(dataloader):
            images = sample_batched['image'].float().cuda()
            labels = sample_batched['label'].cuda()

            inputs = Variable(images)
            targets = Variable(labels)
            # inputs and output shape = [2, 4, 128, 128, 128], targets shape = [2, 128, 128, 128]
            outputs = model(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, targets.long())
            loss.backward()
            optimizer.step()

            # print('(', epoch, ',', i_batch,') loss:', "%.2f" % loss.data.cpu().numpy())
            writer.writerow([epoch, i_batch, "%.2f" % loss.data.cpu().numpy()])

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, '/home/sci/hdai/Projects/UNet/training/epoch_{}_'.format(epoch+700) + 'checkpoint.pth.tar')

        if epoch % 10 == 0:
            torch.save(model, 'MyUNet_{}'.format(epoch+700) + '_in.pkl')

    csvFile.close()

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    model = UNet()
    model.cuda()
    train(model)
    # torch.save(model, 'MyUNet_190_in.pkl')
