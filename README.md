# Este es un modelo que predice la retinopatia diabetica en las 5 fases

## Esta es la forma de como se debe ver el entorno

![Carpetas](imgs/carpeta.png)

## Transformacion de data **_txt2json_**

### Las clases seran guardadas en un **_json_** con el siguiente formato.

```
{
    filenames: [],
    labels: []
}
```

### El nombre de estos JSON estan en el formato por ejemplo 'DDR_train.json'.

### Para poder generarlos se debe correr el **_main.py_** con los siguientes argumentos:

```
python3.8 main.py --txt2json --txt /home/bringascastle/Documentos/datasets-retina/DDR-dataset/DR_grading/valid.txt  --path_src /home/bringascastle/Documentos/datasets-retina/DDR-dataset/DR_grading/valid --save_json ./JSONFiles/DDR --set valid
```

### **_--txt2json_** este aclara que se convertira el json

### Donde **_--txt_** es la ruta del txt con la imagen y la label.

### El txt debe tener el siguiente formato

```
nombredelaimagen etiqueta
Ejemplo:
imagen1.png 0
imagen2.png 1
.
.
.
```

### **_--path_src_** aqui esta la ruta de las imagenes y siempre la ruta va al final sin el '/'

### **_--save_json_** es la ruta donde se va guargar el formato debe ser **_./JSONFiles/{dataset}_** donde dataset se puede poner como en el ejemplo de arriba.

### **_--set_** se define si es **valid** ó **test** ó **train**
