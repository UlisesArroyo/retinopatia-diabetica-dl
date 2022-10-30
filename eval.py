import torch
from torch.utils.data import DataLoader
from data.drdataset import DrDataset
from tqdm import tqdm


def eval(model_load: str, data: str, batch: int, workers: int, device: str, set: str, save: bool = False):

    dataloader = DataLoader(
        DrDataset(data + '{}.json'.format(set), set),
        batch_size=batch,
        num_workers=workers,
    )

    device = torch.device(device)

    checkpoint = torch.load(model_load)
    epoch = checkpoint['epoch'] + 1
    model = checkpoint['model']
    model_str = checkpoint['str']

    print('Modelo {} Epoca: {}'.format(model_str, epoch))

    model.eval()
    process_bar = tqdm(enumerate(dataloader), total=len(dataloader))

    for _, batch in process_bar:
        image, label, f = batch
        label = label.squeeze()

        image = image.to(device)
        label = label.to(device)

        pred = model(image)
        process_bar.set_description_str('Set: {}'.format(), True)
