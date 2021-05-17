# retea echivalenta acestei arhitecturi din Matlab
#https://www.mathworks.com/help/deeplearning/ug/create-simple-deep-learning-network-for-classification.html


import torch
import torch.nn as nn
import time
import copy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Unit(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, str=1, pad=1):
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=ksize, out_channels=out_channels, stride=str,
                              padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        output1 = self.bn(output)
        output2 = self.relu(output1)

        return output2


class basicCNN(torch.nn.Module):
    def __init__(self, nf, num_classes, w, h):  # parametrii si valorile default
        super(basicCNN, self).__init__()
        self.layer1 = Unit(in_channels=3, out_channels=nf)  #wxhx8
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2) # w/2xh/2x8

        self.layer2 = Unit(in_channels=nf, out_channels=2*nf) #w/2 x h/2 x16
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2) # w/4 x h/4 x16

        self.layer3 = Unit(in_channels=2*nf, out_channels= 3*nf)  # w/4 x h/4 x32
        self.fc = nn.Linear(int(w/4 * h/4) * nf*3, num_classes, bias=True)

    def forward(self, input):
        out = self.layer1(input)
        out = self.mp1(out)

        out = self.layer2(out)
        out = self.mp2(out)

        out = self.layer3(out)
        out = out.view(out.size(0), -1)  # Flatten them for FC
        out = self.fc(out)
        return out
        # return F.log_softmax(out)


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, start_epoch=0):
    since = time.time()
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))



        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, val_acc_history
