import json
import torch


class Util():

    def __init__():
        super()

    def save_checkpoint(epoch, model, optimizer, filename):
        state = {
            'epoch': epoch,
            'model': model,
            'optimizer': optimizer
        }

        torch.save(state, filename)

    def clean_loss(filename):
        with open(filename, 'r') as file:
            data = json.load(file)

        data['loss'].clear()

        with open(filename, 'w') as file:
            json.dump(data, file)

    def clean_preds(filename):
        with open(filename, 'r') as file:
            data = json.load(file)

        data['predictions'].clear()

        with open(filename, 'w') as file:
            json.dump(data, file)

    def generarJSON(filename):

        data = {
            "loss": [],
            "predictions": []
        }

        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)

    def guardarLoss(filename, loss):

        with open(filename, 'r') as file:
            data = json.load(file)

        data['loss'].append(loss)

        with open(filename, 'w') as file:
            json.dump(data, file)

    def guardarPrediction(filename, datas):

        with open(filename, 'r') as file:
            data = json.load(file)

        data['predictions'].append(datas)

        with open(filename, 'w') as file:
            json.dump(data, file)
