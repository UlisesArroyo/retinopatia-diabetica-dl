import torch
from torch.utils.data import DataLoader
from data.drdataset import DrDataset
from tqdm import tqdm
#from sklearn.metrics import accuracy_score, precision_score


def eval(model, data: str, batch: int, workers: int, device: str, set: str, save: bool = False):

    dataloader = DataLoader(
        DrDataset(data + '{}.json'.format(set), set),
        batch_size=batch,
        num_workers=workers,
    )

    # device = torch.device(device)

    # checkpoint = torch.load(model_load)
    # epoch = checkpoint['epoch'] + 1
    # model = checkpoint['model']
    # model_str = checkpoint['str']

    # print('Modelo {} Epoca: {}'.format(model_str, epoch))

    # model = model.to(device)
    model.eval()
    process_bar = tqdm(enumerate(dataloader), total=len(dataloader))


    prueba_gt = {0:0, 1:0,2:0,3:0,4:0}
    prueba_p = {0:1, 1:0,2:0,3:0,4:0}

    for _, batch in process_bar:
        image, label, f = batch
        label = label.squeeze()

        image = image.to(device)
        label = label.to(device)

        pred = model(image)

        prueba_gt[int(torch.argmax(pred,dim=1)[0])] += 1
        prueba_p[int(label)] += 1
        process_bar.set_description_str('Set: {}'.format(set), True)

        print(prueba_gt)
        print(prueba_p)