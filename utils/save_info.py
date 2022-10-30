import json
import torch
import csv


class Util():

    def __init__():
        super()

    def txt2json(file, path_src, path_to):

        data = {
            "filenames": [],
            "labels": []
        }

        with open(file, 'r') as arch:
            for info in arch:
                data['filenames'].append(path_src + '/' + info.split(' ')[0])
                data['labels'].append(int(info.split(' ')[1].rstrip('\n')))

        with open(path_to, 'w') as file:
            json.dump(data, file)

        print('Se ha generado el JSON en {}'.format(path_to))

    def csv2json(path_csv, path_src, path_to, columna_i, columna_g, ext=None):

        data = {
            "filenames": [],
            "labels": []
        }

        with open(path_csv, 'r') as file:
            datos = csv.reader(file, delimiter=',')

        next(datos, None)

        for fila in datos:
            img = fila[columna_i]
            grad = fila[columna_g]

            if ext is not None:
                data['filenames'].append(str(path_src + '/' + img))
                data['labels'].append(int(grad))
            else:
                data['filenames'].append(str(path_src + '/' + img + ext))
                data['labels'].append(int(grad))

        with open(path_to, 'w') as file:
            json.dump(data, file)

        print('Se ha generado el JSON en: {}'.format(path_to))

    def save_checkpoint(epoch, model, optimizer, filename, model_str):
        state = {
            'str': model_str,
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
