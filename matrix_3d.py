import numpy as np
from scipy import ndimage
from processLIDC3 import Patient
from tqdm import tqdm
import random
import argparse
import torch
import os
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.graph_objects as go
from ultralytics import YOLO

def calculate_iou(clusters_pred, clusters_ann):
    intersection = np.logical_and(clusters_pred, clusters_ann)

    # Calcular la unión (OR lógico) entre los arrays
    union = np.logical_or(clusters_pred, clusters_ann)

    # Calcular el IoU para cada elemento y luego promediar
    iou = np.sum(intersection) / np.sum(union)
    return iou

def calculate_confusion_matrix(mask_pred, mask_ann):
    print(mask_pred.shape)
    clusters_pred, n_clus_pred = ndimage.label(ndimage.binary_dilation(mask_pred, structure=np.ones((3,3,3))))
    clusters_ann, n_clus_ann = ndimage.label(ndimage.binary_dilation(mask_ann, structure=np.ones((3,3,3))))
    colors = ['red', 'orange', 'yellow', 'green', 'black', 'purple', 'pink', 'brown', 'cyan', 'lime', 'violet', 'gray', 'green', 
              'red', 'orange', 'yellow', 'green', 'black', 'purple', 'pink', 'brown', 'cyan', 'lime', 'violet', 'gray']
    fig = go.Figure()
    for i in range(n_clus_ann):
        i= i +1
        condition_indices = np.where(clusters_ann == i)
        # Obtener los valores y los índices que cumplen la condición
        selected_indices = np.array(condition_indices).T  # Transponer los índices para obtener (n_puntos, 3)
        # Crear un scatter plot utilizando los índices como coordenadas
        

        # # # # Agregar traza para los puntos con los índices originales
        fig.add_trace(go.Scatter3d(
            x=selected_indices[:, 0],
            y=selected_indices[:, 1],
            z=selected_indices[:, 2],
            mode='markers',
            marker=dict(
                size=4,  # Cambiamos el tamaño a 4 (puedes ajustar este valor)
                color=colors[i-1],  
                opacity=0.8
        ),
            name=f'Cluster: {i}'
        ))
    condition_indices_pred = np.where(clusters_pred > 0.5)
    selected_indices_pred = np.array(condition_indices_pred).T 
    fig.add_trace(go.Scatter3d(
        x=selected_indices_pred[:, 0]+0.2,
        y=selected_indices_pred[:, 1]+0.2,
        z=selected_indices_pred[:, 2]+0.2,
        mode='markers',
        marker=dict(
            symbol='cross', 
            size=4,  # Cambiamos el tamaño a 4 (puedes ajustar este valor)
            color='blue',  
            opacity=0.8
    ),
        name='Conjunto Predicho'
    ))
    # # Configurar el diseño del gráfico
    fig.update_layout(scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z'
    ))

    # Mostrar el gráfico
    fig.show()
    
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    
    # print('n_clus_pred', n_clus_pred)
    # print('n_clus_ann', n_clus_ann)
    # print(contours_a)
    if n_clus_pred ==0:
        false_negative += n_clus_ann
        return {
        'TP': true_positive,
        'TN': true_negative,
        'FP': false_positive,
        'FN': false_negative
    }
    if n_clus_ann==0:
        false_positive += n_clus_pred
        return {
        'TP': true_positive,
        'TN': true_negative,
        'FP': false_positive,
        'FN': false_negative
    }
    pred_pequenios = [] 
    for i_clus_pred in range(n_clus_pred):
        i_clus_pred+=1
        if 40>np.sum(clusters_pred == i_clus_pred):
            pred_pequenios.append(i_clus_pred)
    # print('\n pred_pequenios', pred_pequenios)
    n_pred_intersect_total = []
    for i_clus_ann in range(n_clus_ann):
        i_clus_ann = i_clus_ann+1
        n_pred_intersect = []
        for i_clus_pred in range(n_clus_pred):
            i_clus_pred = i_clus_pred+1 
            if i_clus_pred in pred_pequenios:
                continue
            iou = calculate_iou(np.where(clusters_pred == i_clus_pred, 1, 0), np.where(clusters_ann == i_clus_ann, 1, 0))
            if iou >0:
                n_pred_intersect.append(i_clus_pred)
                n_pred_intersect_total.append(i_clus_pred)
        iou = calculate_iou(np.where(np.isin(clusters_pred, n_pred_intersect), 1, 0), np.where(clusters_ann == i_clus_ann, 1, 0))
        if iou > IOU_THRESHOLD:
            true_positive+=1
        else:
            # print('\n', 'holaa: ', iou, '\n')
            false_negative+=1

    pred_clusses_no_iou = [i for i in range(n_clus_pred) if not i in n_pred_intersect_total and not i in pred_pequenios]
    false_positive += len(pred_clusses_no_iou)


    confusion_matrix = {
        'TP': true_positive,
        'TN': 0, #true_negative,
        'FP': false_positive,
        'FN': false_negative
    }
    # print(confusion_matrix)

    return confusion_matrix


def get_arrays(id_patient):
    patient = Patient(id_patient)
    patient.scale()
    images, mask = patient.get_tensors(scaled = True)
    mask = mask.cpu().detach().numpy()
    images = images.cpu().detach().numpy()
    slices = []
    for i in range(mask.shape[0]):
        if np.all(mask[i,0] == 0):
            continue
        slices.append(i)
    mask_predicciones = []
    mask_labels = []
    if len(slices)==0:
        return np.array([[[0], [0]], [[0], [0]]]), np.array([[[0], [0]], [[0], [0]]])
    for s in tqdm(slices):
        img_path = f'{args.val_path}{id_patient}_{s}.png'
        prediction = model.predict(img_path, conf=THRESHOLD)
        if prediction[0].masks is not None:
            prediccion = np.max(prediction[0].masks.data.cpu().numpy(), axis=0)
            print('\n \n \n \n no es none \n \n \n \n')
        else:
            prediccion = np.zeros(mask[s,0].shape)
            
        mask_predicciones.append(prediccion)
        mask_labels.append( mask[s,0])
    # if np.all(mask_labels==0):
    #     return np.array([[[0], [0]], [[0], [0]]]), np.array([[[0], [0]], [[0], [0]]])

    arrayyy = np.stack(mask_labels)
    arrayyy_pred = np.stack(mask_predicciones)
    if np.all(arrayyy_pred==0):
        print('\n \n \n \n es todo 0!! \n \n \n \n')
    condition_indices = np.where(arrayyy > 0.5)
    condition_indices_pred = np.where(arrayyy_pred > 0.5)
    # Obtener los valores y los índices que cumplen la condición
    selected_indices = np.array(condition_indices).T  # Transponer los índices para obtener (n_puntos, 3)
    selected_indices_pred = np.array(condition_indices_pred).T  # Transponer los índices para obtener (n_puntos, 3)
    print(selected_indices_pred)
    # Crear un scatter plot utilizando los índices como coordenadas
    fig = go.Figure()

    # # # # Agregar traza para los puntos con los índices originales
    fig.add_trace(go.Scatter3d(
        x=selected_indices[:, 0],
        y=selected_indices[:, 1],
        z=selected_indices[:, 2],
        mode='markers',
        marker=dict(
            size=4,  # Cambiamos el tamaño a 4 (puedes ajustar este valor)
            color='red',  
            opacity=0.8
    ),
        name='Conjunto Original'
    ))

    # Agregar traza para los puntos con los índices predichos
    fig.add_trace(go.Scatter3d(
        x=selected_indices_pred[:, 0]+0.2,
        y=selected_indices_pred[:, 1]+0.2,
        z=selected_indices_pred[:, 2]+0.2,
        mode='markers',
        marker=dict(
            symbol='cross', 
            size=4,  # Cambiamos el tamaño a 4 (puedes ajustar este valor)
            color='blue',  
            opacity=0.8
    ),
        name='Conjunto Predicho'
    ))

    # # Configurar el diseño del gráfico
    fig.update_layout(scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z'
    ))

    # Mostrar el gráfico
    fig.show()
    return np.stack(mask_predicciones), np.stack(mask_labels)

def sumar_diccionarios(dic1, dic2):
    for key, value in dic2.items():
        if key in dic1:
            dic1[key] += value
        else:
            dic1[key] = value
    return dic1

def get_confusion_matrix2(patients_list):
    confusion_matrix_total = {
        'TP': 0,
        'TN': 0,
        'FP': 0,
        'FN': 0
    }
    print('patients_list', patients_list)
    for patient_id in tqdm(patients_list):
        print(patient_id)
        mask_predicciones, mask_labels = get_arrays(patient_id)
        # print(mask_predicciones.shape, mask_labels.shape)
        if np.all(mask_labels==0):
            continue
        mat =calculate_confusion_matrix(mask_predicciones, mask_labels)
        confusion_matrix_total = sumar_diccionarios(confusion_matrix_total, mat)
    return confusion_matrix_total

def plot_confusion_matrix(confusion_dict, save= './',show=False, threshold=999):
    labels = list(confusion_dict.keys())
    confusion_matrix = [[confusion_dict['TP'], confusion_dict['FP']],
                        [confusion_dict['FN'], confusion_dict['TN']]]

    # plt.figure(figsize=(6, 4))
    fig, ax = plt.subplots()

    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Reds", xticklabels=('label: nodulo', 'label: no_nodulo'), yticklabels=('pred: nodulo', 'pred: no_nodulo'))
    ax.set_xlabel("Label")
    ax.set_ylabel("Prediccion")
    ax.set_title(f"Confusion Matrix: IoU. threshold= {THRESHOLD}, iou_threshold = {IOU_THRESHOLD}")
    if show:
        # Mostrar la figura
        plt.show()
    if save is not None:
        fecha_actual = datetime.now()
        # Formatear la fecha en el formato deseado (por ejemplo, "año_mes_dia_hora_minuto_segundo")
        fecha = fecha_actual.strftime("%Y-%m-%d_%H-%M-%S")
        path = save+f'confusion_matrix_iou_{fecha}.png'
        plt.savefig(path, dpi=300)
        print(f'figura guardada {path}')



# array1 = np.array([[[0, 1, 1, 0,0,1,0,0,0,1,1,0,0,0,1],
#                      [0, 1, 1, 1,0,0,0,0,0,1,1,1,0,0,0],
#                      [0, 0, 0, 1,0,0,0,0,0,0,0,1,1,0,0]],
#                     [
#                     [0, 1, 1, 0,0,0,0,0,0,1,1,0,0,0,1],
#                      [0, 1, 1, 1,0,0,1,0,0,1,1,1,0,0,0],
#                      [0, 0, 0, 1,0,0,1,0,0,0,0,1,1,0,0]
#                     ]
#                    ])
# array2 = np.array([[[0, 0, 0, 0,0,1,0,0,0,1,1,0,0,0,1],
#                      [0, 1, 1, 1,0,0,0,0,0,1,1,1,0,0,0],
#                      [0, 0, 0, 1,0,0,0,0,0,0,0,1,1,0,0]],
#                     [
#                     [0, 1, 1, 0,0,0,0,0,0,1,1,0,0,0,1],
#                      [0, 1, 1, 1,0,0,1,0,0,1,1,1,0,0,0],
#                      [0, 0, 0, 1,0,0,1,0,0,0,0,1,1,0,0]
#                     ]
#                    ])



random.seed(123)
parser = argparse.ArgumentParser()
# Agregar los argumentos necesarios
parser.add_argument('--val', action='store_true', default = True)
parser.add_argument('--model', type=str, default='./default_model.pt')
parser.add_argument('--save', type=str, default='./')
parser.add_argument('--val_path', type=str, default='./')
parser.add_argument('--threshold', type=float, default=0.2)
parser.add_argument('--iou_threshold', type=float, default=0.5)
# parser.add_argument('--batch', type=float, default=5)
args = parser.parse_args()

print('Buscando los pacientes...', flush= True)
img_names = os.listdir(args.val_path)
# archivo = open('./failed_patients.txt', 'r')  # Reemplaza 'nombre_archivo.txt' por el nombre de tu archivo
# failed_patients = []

# for linea in archivo:
#     linea = linea.strip()  # Elimina los espacios en blanco al principio y al final de la línea
#     failed_patients.append(linea)
# archivo.close()
patients = set([pat.split('_')[0]for pat in img_names])

# n_val = int(len(patients) * args.valsplit)
# if args.val:
#     # Seleciona aleatoriamente el 30% de los patients
#     patients_list = random.sample(patients, n_val)
# else:
#     val_patients = random.sample(patients, n_val)
#     patients_list = [nombre for nombre in patients if nombre not in val_patients]
    
# MODEL_PATH = '/home/faraujo/TFM/yolo_trainings/train_100_include02_large/best.pt'

model = YOLO(args.model)
# patients_list = ['LIDC-IDRI-0186', 'LIDC-IDRI-0001', 'LIDC-IDRI-0002', 'LIDC-IDRI-0013', 'LIDC-IDRI-0170', 'LIDC-IDRI-0191']
THRESHOLD = args.threshold
IOU_THRESHOLD = args.iou_threshold
# patients_list = ['LIDC-IDRI-0604']
mat = get_confusion_matrix2(patients)
plot_confusion_matrix(mat, save= args.save)
    
# print(mat)























