import torch
from torch.utils.data import DataLoader
from data.drdataset import DrDataset
from models.resnet101 import resNet101
from models.convnext import convNextSmall
from tqdm import tqdm
from utils.save_info import Util


def train(model_str, model_load, json_result, dump, data, epochs, lr, decay_lr,
          batch, workers, momentum, weigth_decay, device):

    dataloader_train = DataLoader(
        DrDataset(data + 'train.json', 'train'),
        batch_size=batch,
        num_workers=workers,
    )

    dataloader_valid = DataLoader(
        DrDataset(data + 'valid.json', 'valid'),
        batch_size=batch,
        num_workers=workers
    )

    device = torch.device(device)
    classes = 5
    model = None

    if model_load is None:
        start_epoch = 0
        if model_str == 'resnet':
            model = resNet101(classes)
        else:
            model = convNextSmall(classes)

        optimizer = torch.optim.Adam(
            model.parameters(), lr, weight_decay=weigth_decay)

    else:
        checkpoint = torch.load(model_load)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(start_epoch, epochs):
        train_one_epoch(model, dataloader_train, optimizer,
                        criterion, epoch, device, json_result
                        )

        Util.save_checkpoint(epoch, model,optimizer, dump)


def train_one_epoch(model, dataloader, optimizer: torch.optim.Adam, criterion, epoch, device, json_result):

    model.train()
    process_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    loss_total = 0.0

    for _, batch in process_bar:
        image, label, f = batch
        label = label.squeeze()

        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        pred = model(image)

        loss = criterion(pred, label)
        loss_total += float(loss)
        loss.backward()
        optimizer.step()
        process_bar.set_description_str(
            'Epoch {} : Loss: {:.3f}'.format(epoch, float(loss)), True)

    Util.guardarLoss(json_result, loss_total/len(dataloader))
    print('Perdida promedio: {}'.format(loss_total/len(dataloader)))
