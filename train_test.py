
import torch.nn.functional as F
import numpy as np
import torch

def train_single_epoch(model, optimizer, train_loader, device=None):

    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_size = len(train_loader.dataset)

    total_loss = 0.
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        loss = F.cross_entropy(output, labels, reduction='sum')
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # predict = torch.zeros_like(labels, requires_grad=False)
        # predict[output > 0.5] = 1
        # n_correct += (predict == labels).sum().item() / float(output.size(0))

    return total_loss / float(dataset_size)

def test(model, test_loader, device=None):

    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_size = len(test_loader.dataset)

    with torch.no_grad():
        total_loss = 0.
        n_correct = 0.
        model.eval()
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            total_loss += F.cross_entropy(output, labels.float(), reduction='sum').item()
            predict = torch.argmax(output, -1)
            n_correct += (predict == labels).sum().item()

    avg_loss = total_loss / float(dataset_size)
    accuracy = n_correct / float(dataset_size)

    return avg_loss, accuracy


def train(model, optimizer, max_epoch, train_loader,
          val_loader=None, checkpoint_dir=None, max_tolerance=-1,
          device=None):

    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_loss = 99999.
    tolerated = 0

    log = np.zeros([max_epoch, 2], dtype=np.float)

    for e in range(max_epoch):

        log[e, 0] = train_single_epoch(model, optimizer, train_loader, device)

        print('Batch #{:d}'.format(e+1))
        print('Train Loss: {:.3f}'.format(log[e, 0]))

        if val_loader is not None:

            log[e, 1], log[e, 2] = test(model, val_loader, device)

            print('Val Loss: {:.3f}'.format(log[e, 1]))
            print('Val Accs: {:.3f}'.format(log[e, 2]))

            if (best_loss > log[e, 2]):
                best_loss = log[e, 2]
                if not os.path.exists(args.save):
                    os.makedirs(args.save)
                checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint'+str(e+1)+'.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print('Best Loss! Saved.')
            elif max_tolerance < 0:
                tolerated += 1
                if tolerated > max_tolerance:
                    return log[0:e, :]

    return log

if __name__ == '__main__':

    import argparse
    import data
    from torch.utils.data import random_split, DataLoader
    from torch.optim import Adam
    import model

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--n_epoch', type=int, required=True)
    parser.add_argument('--save', type=str, required=True)
    args = parser.parse_args()

    dataset = data.WasteNetDataset(args.data_dir, 'none')
    trainset, valset, testset = dataset.split([0.7, 0.1, 0.2])
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, drop_last=True)
    valloader = DataLoader(valset, batch_size=64, shuffle=False)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = model.WasteNet().to(device)
    if args.weights != None:
        net.load_state_dict(torch.load(args.weights, map_location=device))
    optimizer = Adam(net.parameters(), lr=1e-4)
    train(net, optimizer, args.n_epoch, trainloader, valloader, args.save, -1, device)

    torch.save(net.state_dict(), os.path.join(args.save, 'final.pth'))
