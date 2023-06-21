# yolo-lidc
Script para convertir el dataset LIDC-IDRI (imagenes .dicom + anotaciones de segmentacion de 4 radiologos) a formato YOLO, en contreto para yolov8-segmentation.



# Qué se obtiene

Estructura de carpetas de dataset en formato yolo:
```
.
├── train
│   ├── images
│   └── labels
├── train.yaml
└── validation
    ├── images
    └── labels
```

El `train.yaml` incluye:

```
names:
- nodulo

nc: 1

train: ./train/images/
val: ./validation/images/
```

Las imagenes son guardads en formato `.png` siendo cada slice de un TC de un paciente una imagen, guardadas en `images`.

Las etiquetas siguen formato YOLO, es decir, un `.txt` por cada imagen con el mismo nombre las carpetas `labels`. Estas siguen el formato:

n_clase x1 y1 x2 y2 x3 y3 x4 y4 x5 y5 ...

Ejemplo: 0 0.324 0.435 0.435 0.5474 0.45754 0.4456 0.45645 0.23462 0.8644 0.23423

- n_clase = 0 ya que solo hay una clase (yolo es segmentador de instacias)

- xi yi son puntos que perfilan el nodulo en relativo intervalo de valores.(0,1)



El nombre de cada imagen y label sigue el criterio:

id_paciente+n_slice

Ejemplo: 
 - Slice numero 143 del paciente con id LIDC-IDRI-0957:
  `LIDC-IDRI-0957_143.png`  y  `LIDC-IDRI-0957_143.txt`


# Usage

Revisar el `yolodataset.py`:

```
if __name__ =='__main__':
    path2newdataset='/path/to/new/dataset_yolo'
    path2olddataset = '/path/to/old/LIDC-IDRI_dataset'

    create_dataset(path2olddataset, path2newdataset, val=0.2, percent_include=0.2)
```

Parametros: 
- `val`: porcentaje de pacientes que iran a validation
- `percent_include`: porcentaje de las imagenes que no tienen ningun nodulo etiquetado que serán consideradas.

`/path/to/new/dataset_yolo` debe tener esta pinta:
```
.
├── train
│   ├── images
│   └── labels
├── train.yaml
└── validation
    ├── images
    └── labels
```

y `/path/to/old/LIDC-IDRI_dataset` debe tener esta pinta:
```
.
├── LIDC-IDRI-0002
├── LIDC-IDRI-0005
├── LIDC-IDRI-0013
├── LIDC-IDRI-0055
:
└── LIDC-IDRI-0129
```



