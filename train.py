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
          batch_t, batch_s, workers_t, workers_s, momentum, weigth_decay, devices):

    dataloader_train = DataLoader(
        DrDataset(data + 'train.json', 'train'),
        batch_size=batch_t,
        num_workers=workers_t,
        shuffle=True
    )

    device = torch.device(devices)
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
        checkpoint = torch.load(model_load, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        for g in optimizer.param_groups:
            g['lr'] = lr

    criterion = torch.nn.CrossEntropyLoss()
    model = model.to(device)
    data_eval = './JSONFiles/DDR/DDR_'

    best = 0.0
    best_dump = ''

    acc_conteo = []
    acc = 0.0

    factor_lr = decay_lr

    for epoch in range(start_epoch, epochs):

        if len(acc_conteo) == 5 and epoch > 30:
            if average(acc_conteo) <= acc:
                print('Decay lr...')
                adjust_learning_rate(optimizer, factor_lr)
                acc_conteo.clear()
            else:
                acc_conteo.clear()
        else:
            if acc != 0.0:
                acc_conteo.append(acc)

        train_one_epoch(model, dataloader_train, optimizer,
                        criterion, epoch, device, json_result
                        )

        Util.save_checkpoint(epoch, model, optimizer, dump, model_str)
        print('Evaluando....')
        acc, aps = eval(model, data_eval, batch_s,
                        workers_s, device, 'valid', False)

        Util.saveInfoXepoch(os.path.dirname(json_result) +
                            '/info_train_{}.json'.format(model_str), epoch, acc, aps)

        if best < acc:
            best_dump = os.path.dirname(json_result) + \
                '/{}_best.pth'.format(model_str)
            best = acc
            Util.save_checkpoint(epoch, model, optimizer, best_dump, model_str)


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


def adjust_learning_rate(optimizer, scale):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale


def average(lst):
    return sum(lst) / len(lst)
