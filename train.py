import torch
from torch.utils.data import DataLoader
from data.drdataset import DrDataset
from models.resnet101 import resNet101
from models.convnext import convNextSmall
from tqdm import tqdm
from utils.save_info import Util
from eval import eval
import os


def train(model_str, model_load, json_result, dump: str, data, epochs, lr, decay_lr,
          batch_t, batch_s, workers_t, workers_s, momentum, weigth_decay, device):

    dataloader_train = DataLoader(
        DrDataset(data + 'train.json', 'train'),
        batch_size=batch_t,
        num_workers=workers_t,
        shuffle=True
    )

    device = torch.device(device)
    classes = 5
    model = None

    if model_load is None:
        start_epoch = 0

        if model_str == 'resnet':
            model = resNet101(classes)

        if model_str == 'convnext':
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

    data_eval = './JSONFiles/DDR/DDR_'
    best = 0.0
    for epoch in range(start_epoch, epochs):

        train_one_epoch(model, dataloader_train, optimizer,
                        criterion, epoch, device, json_result
                        )

        Util.save_checkpoint(epoch, model, optimizer, dump, model_str)

        acc, aps = eval(model, data_eval, batch_s,
                        workers_s, device, 'valid', False)

        Util.saveInfoXepoch(os.path.dirname(json_result) +
                            '/info_train_{}.json'.format(model_str), epoch, acc, aps)

        if best < acc:
            dump = dump.split('.')
            dump = dump[0] + '_best' + '.pth'
            Util.save_checkpoint(epoch, model, optimizer, dump, model_str)


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
            'Epoch {} : Loss: {:.3f}'.format(epoch + 1, float(loss)), True)

    Util.guardarLoss(json_result, loss_total/len(dataloader))
    print('Perdida promedio: {:.4f}'.format(loss_total/len(dataloader)))
