from processLIDC import Patient
import cv2
import numpy as np
# import matplotlib.pyplot as plt
import random
import os
from tqdm import tqdm




def dicom2jpg(id_patient, path2newdataset = './dataset'):
    patient = Patient(id_patient)
    patient.scale()
    h, w, n_slices = np.shape(patient.imgs_scaled)
    for i in range(n_slices):
        fullpath = f'{path2newdataset}/images/{id_patient}_{i}.png'
        imagen_8bits = np.interp(patient.imgs_scaled[:,:,i], (0, 8), (0, 255)).astype(np.uint8)
        cv2.imwrite(fullpath, imagen_8bits)


def ann2yolo(id_patient, path2newdataset = './dataset'):
    patient = Patient(id_patient)
    h, w, n_slices = np.shape(patient.mask)
    no_tumor = 0
    tumor =0
    for i in range(n_slices):
        fullpath = f'{path2newdataset}/labels/{id_patient}_{i}.txt'
        mask = np.interp(patient.mask[:,:,i], (0, 1), (0, 255)).astype(np.uint8)
        _, threshold = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) ==0:
            no_tumor+=1
        else:
            tumor+=1
        # Crear un archivo de texto para escribir los contornos
        with open(fullpath, "w") as file:
            for contour in contours:
                # Escribir '0' al principio de cada línea
                file.write("0")

                # Obtener las coordenadas de los puntos del contorno
                points = contour.reshape(-1, 2)

                # Escribir las coordenadas en el archivo
                for point in points:
                    x, y = point
                    file.write(f" {x/w} {y/h}")

                # Nueva línea para el siguiente contorno
                file.write("\n")
    return tumor, no_tumor


def create_dataset(path2olddataset,path2newdataset, val = 0.2):
    print('convirtiendo dataset a yolo...')
    id_patients = os.listdir(path2olddataset)
    failed_patients = []
    try:
        archivo = open('./failed_patients.txt', 'r')  # Reemplaza 'nombre_archivo.txt' por el nombre de tu archivo
        for linea in archivo:
            linea = linea.strip()  # Elimina los espacios en blanco al principio y al final de la línea
            failed_patients.append(linea)
        archivo.close()
    except:
        print('No se han tomado failed patients')
    id_patients = [id_patient for id_patient in id_patients if id_patient not in failed_patients]
    tumor_train, no_tumor_train = 0, 0
    tumor_val, no_tumor_val = 0, 0
    for id_patient in tqdm(id_patients):
        train_random = random.random() > val
        if train_random:
            dicom2jpg(id_patient, path2newdataset+'/train')
            tumor, no_tumor = ann2yolo(id_patient, path2newdataset+'/train')
            tumor_train += tumor
            no_tumor_train += no_tumor
        else:
            dicom2jpg(id_patient, path2newdataset+'/validation')
            tumor, no_tumor = ann2yolo(id_patient, path2newdataset+'/validation')
            tumor_val += tumor
            no_tumor_val += no_tumor

    print('train: \n','\t images:', len(os.listdir(path2newdataset+'/train/images')), '\t labels:', len(os.listdir(path2newdataset+'/train/labels')))
    print('validation: \n', '\t images:',len(os.listdir(path2newdataset+'/validation/images')), '\t labels:', len(os.listdir(path2newdataset+'/validation/labels')))
    print(f'train: \n \t Con nodulo: {tumor_train} \t Sin nodulo: {no_tumor_train}')
    print(f'validation: \n \t Con nodulo: {tumor_val} \t Sin nodulo: {no_tumor_val}')


if __name__ =='__main__':
    path2newdataset='/home/faraujo/TFM/datasetPNG'

    path2olddataset = '/home/faraujo/TFM/manifest-1675801116903/LIDC-IDRI'
    
    create_dataset(path2olddataset, path2newdataset, val=0.2)