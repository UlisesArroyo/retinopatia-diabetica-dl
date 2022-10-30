import argparse
import os
from sys import exit
from eval import eval
from train import train
from utils.save_info import Util


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')

    # Acciones para el modelo
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--model', default=None)
    parser.add_argument('--eval', action='store_true', default=False)

    # Acciones para el dataset
    parser.add_argument('--csv2json', action='store_true', default=False)
    parser.add_argument('--txt2json', action='store_true', default=False)

    # Hyperparametros
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--decay_lr', type=float, default=0.1)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weigth_decay', type=float, default=5e-4)
    parser.add_argument('--device', type=int, default=0)

    # Ubicaciones de archivos
    parser.add_argument('--load_model', default=None)
    parser.add_argument('--dataloader_json', default=None)
    parser.add_argument('--json_result', default=None)
    parser.add_argument('--dump', default=None)

    args = parser.parse_args()

    if not os.path.exists('./runs'):
        os.makedirs('./runs', exist_ok=True)

    if args.train:

        if args.model is None or args.model not in ['resnet', 'convnext']:
            print('Elige un modelo a entrenar')
            exit()

        if args.json_result is not None:
            json_result = args.json_result
            if not os.path.exists(args.json_result):
                Util.generarJSON(json_result)

        if args.dump is not None:
            dump = args.dump
            print('Elije donde guardar tu modelo')
            exit()

        if args.dataloader_json is not None:
            dataloader_json = args.dataloader_json
            print('Ingresa tu JSON de tu base de datos')
            exit()

        epoch, lr, decay_lr, batch, workers, momentum, weigth_decay, device = args.epochs, args.lr, args.decay_lr, args.batch, args.workers, args.momentum, args.weigth_decay, args.device

        model_load = args.load_model

        train(args.model, model_load, json_result, dump,
              dataloader_json, epoch, lr, decay_lr, batch, batch,
              workers, workers, momentum, weigth_decay, device)
