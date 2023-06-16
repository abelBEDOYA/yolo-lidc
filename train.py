import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import random
import argparse
import time
from collections import deque
# Mi libreria:
from processLIDC import Patient
import datetime
import cv2
random.seed(123)

def train_val_split(patients_list, val_split):
    """TOma la lsita de pacientes list(str) y hace la separacion
    en train y validation segun la proporcion indicada en val_split.
    Args:
        patients_list (list(str)): lista on los id de los pacientes
        val_split (float): proporcion destinada a validation
    Returns:
        train_patients, val_val_patients (list, list): lista de
            nombres de pacientes para train y validation
        """
    n_val = int(len(patients_list) * val_split)

    # Seleciona aleatoriamente el 30% de los patients
    val_patients = random.sample(patients_list, n_val)

    # Crea una lista con los patient que no fueron seleccionados
    train_patients = [nombre for nombre in patients_list if nombre not in val_patients]
    return train_patients, val_patients


def get_val_loss(model, val_patients, batch_size=4, loss_type = 1):
    if len(val_patients)==0:
        return 0

    loss_batch = np.array([])
    batch_loss_history = np.array([])
    loss_patient = np.array([])
    print('Realizando validacion...')
    tqdm_val_patients = tqdm(val_patients,leave=False, position=0)
    for id_pat in tqdm_val_patients:
        time.sleep(1)
        
        tqdm_val_patients.set_description('{}. {}. Progreso val.:'.format(get_tiempo(),id_pat))
        
        # Cargamos datos de un paciente:
        patient = Patient(id_pat)
        
        # Escalamos:
        patient.scale()
        
        # Obtenemos los tensores:
        imgs, mask = patient.get_tensors(scaled=True)
        if torch.cuda.is_available():
            device = torch.device('cuda')
            imgs, mask = imgs.to(device), mask.to(device)
        # Preparamos tensores para recorrerlos:
        # primera = 2
        # ultima = 10
        # dataset = TensorDataset(imgs[primera:ultima], mask[primera:ultima])
        dataset = TensorDataset(imgs, mask)
        

        train_loader = DataLoader(dataset,batch_size=batch_size, shuffle=True)
        loss_batch = np.array([])
        for batch_idx, (data, target) in enumerate(train_loader):

            # # Forward pass
            output = model(data)
            # Calcular pérdida
            loss = loss_function(output[:,0], target, loss_type = loss_type)
            loss_batch = np.append(loss_batch, loss.item())
            batch_loss_history = np.append(batch_loss_history, loss.item())

        loss_patient = np.append(loss_patient, np.mean(np.array(loss_batch)))
    val_mean_loss = np.mean(loss_patient)
    return val_mean_loss


def plot(data, show=False, path_save=None, name_plot='loss_plot'):
    epoch_loss_history = data['epoch_loss_history']
    batch_loss_history = data['batch_loss_history']
    patient_loss_history = data['patient_loss_history']
    epoch_val_loss_history = data['epoch_val_loss_history']
    n_epochs = len(epoch_loss_history)
    #plt.plot(np.linspace(1, n_epochs, np.array(batch_loss_history).shape[0]), np.log(np.array(batch_loss_history)), label='Train Batch Loss')
    plt.plot(np.linspace(1, n_epochs, np.array(patient_loss_history).shape[0]), np.log(np.array(patient_loss_history)), label='Train Patient Loss')
    plt.plot(np.linspace(1, n_epochs, n_epochs), np.log(np.array(epoch_loss_history)), label='Train Epoch Loss')
    plt.plot(np.linspace(1, n_epochs, n_epochs), np.log(np.array(epoch_val_loss_history)), label='Val. Epoch Loss')
    plt.title('Loss: Binary Cross Entropy')
    plt.xlabel('Epoch')
    plt.ylabel('log(loss)')
    plt.legend(loc = 'best', frameon=True)
    if path_save is not None:
        plt.savefig(path_save+'{}.png'.format(name_plot), dpi=300)
        print('Loss plots guardados')
    if show:
        plt.show()
    plt.close('all')


def save_model(model, path='./', model_name='model', extension = '.pt'):
    if extension in ['.pt', '.pth']:
        pass
    else:
        extension = '.pt'
        
    if path[-1]=='/':
        pass
    else:
        path = path+'/'
        
    # Guardar el modelo
    if extension == '.pt':
        model_scripted = torch.jit.script(model) # Export to TorchScript
        model_scripted.save(path+model_name+'.pt') # Save
    else:
        torch.save(model.state_dict(), path+model_name+'.pth')
    print('Modelo {}{}.pth guardado.'.format(path, model_name))


def get_tiempo():
    fecha_hora_actual = datetime.datetime.now()
    # Obtener partes individuales de la fecha y hora
    anio = fecha_hora_actual.year
    mes = fecha_hora_actual.month
    dia = fecha_hora_actual.day
    hora = fecha_hora_actual.hour
    minuto = fecha_hora_actual.minute
    segundo = fecha_hora_actual.second
    tiempo = '{}-{}-{}. {}:{}:{}'.format(anio, mes, dia, hora, minuto, round(segundo))
    return tiempo


def loss_function(output, target, loss_type = 1):
    if loss_type == 1:
        weight_zero = 1
        weight_one = 200
        # Definir función de pérdida
        loss_fn = nn.BCELoss()
        loss_ = loss_fn(output, target)
        one_pixels = torch.eq(target, 1).float()
        zero_pixels = torch.eq(target, 0).float()
        weighted_loss = (weight_one* one_pixels * loss_) + (weight_zero * zero_pixels * loss_)
        scalar_loss = torch.mean(weighted_loss)  # Aplicar reducción para obtener un escalar
        return scalar_loss
    elif loss_type == 2:
        intersection = torch.sum(output * target)
        dice_coefficient = (2 * intersection) / (torch.sum(output) + torch.sum(target) + 1e-7)
        loss_dice = 1 - dice_coefficient
        return loss_dice
    elif loss_type == 3:
        intersection = torch.sum(output * target)
        union = torch.sum(output) + torch.sum(target) - intersection
        iou = intersection / (union + 1e-7)  # small constant to avoid division by zero
        loss_iou = 1 - iou
        return loss_iou
    elif loss_type == 4:
        img = output.cpu().detach().numpy()[0]
        # cv2.imshow('output', img*255)
        # cv2.waitKey(0)
        loss_abs = torch.abs(output - target)
        return torch.mean(loss_abs)
    else:
        print('Indica una loss function que sea 1, 2 o 3. Has indicado loss = {}'.format(loss))



def train(model, n_epochs:int =4, 
          batch_size: int = 4, 
          val_split: float = 0.2,
          path2dataset: str = '../../manifest-1675801116903/LIDC-IDRI/',
          path2savefiles: str = './',
          plot_metrics: bool = False,
          save_plots: bool = False,
          save_epochs = None,
          model_extension = '.pt',
          failed_patients: list = [],
          loss_type: int = 1):
    """Ejecuta el entrenamiento

    Args:
        model (_type_): modelo a entrenar
        epochs (int, optional): numero de epocas. Defaults to 4.
        batch_size (int, optional): batch de imagenes (no pacientes) a evaluar antes de haer backprop. Defaults to 4.
        val_split (float, optional): porcentaje del dataset a validation. Defaults to 0.2.
        path2dataset: str = '../../manifest-1675801116903/LIDC-IDRI/',
        plot: bool = False,
        save_plots: bool = False)
    """
    patients = os.listdir(path2dataset)
    patients = [pat for pat in patients if not pat=='LICENSE' and pat not in failed_patients]

    train_patients, val_patients = train_val_split(patients, val_split)

    # Definir optimizador
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    loss_batch = np.array([])
    batch_loss_history = np.array([])

    loss_patient = np.array([])
    patient_loss_history = np.array([])

    epoch_loss_history = np.array([])
    epoch_loss_history = np.array([])
    epoch_val_loss_history = np.array([])
    tiempos_paciente = deque([6,6,6,6,6], maxlen=5)
    print('Inicio de entrenamiento: {}'.format(get_tiempo()))
    for epoch in range(n_epochs):
        print(f'Epoch: {epoch+1}/{n_epochs}')
        loss_patient = np.array([])
        random.shuffle(train_patients)
        len_train_patients = len(train_patients)
        tqdm_train_patients = tqdm(train_patients,leave=False, position=0)
        for i, id_pat in enumerate(tqdm_train_patients):
            inicio = time.time()
            time.sleep(1)
            
            tqdm_train_patients.set_description('{}. Rate {} s/p. {}/{}. {}. Progreso de la epoca:'.format(get_tiempo(), 
                                                                                                           round(sum(tiempos_paciente)/5, 2),
                                                                                                           i,
                                                                                                           len_train_patients,
                                                                                                           id_pat))
            # Cargamos datos de un paciente:
            patient = Patient(id_pat)

            # Escalamos:
            patient.scale()

            # Obtenemos los tensores:
            imgs, mask = patient.get_tensors(scaled=True)
            if torch.cuda.is_available():
                device = torch.device('cuda')
                imgs, mask = imgs.to(device), mask.to(device)

            # Preparamos tensores para recorrerlos:
            # primera = 2
            # ultima = 10
            # dataset = TensorDataset(imgs[primera:ultima], mask[primera:ultima])
            dataset = TensorDataset(imgs, mask)

            train_loader = DataLoader(dataset,batch_size=batch_size, shuffle=True)
            loss_batch = np.array([])
            for batch_idx, (data, target) in enumerate(train_loader):
                if torch.mean(target)==0:
                    print('es 0')
                    continue
                print(torch.mean(target))
                # # Forward pass
                output = model(data)
                # Calcular pérdida
                loss = loss_function(output[:,0], target, loss_type=loss_type)

                # # # Calcular gradientes y actualizar parámetros
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                loss_batch = np.append(loss_batch, loss.item())
                batch_loss_history = np.append(batch_loss_history, loss.item())
            del data
            del target
            del dataset
            del patient
            loss_patient = np.append(loss_patient, np.mean(np.array(loss_batch)))
            patient_loss_history = np.append(patient_loss_history, np.mean(np.array(loss_batch)))
            tiempo_paciente = time.time()-inicio
            tiempos_paciente.append(tiempo_paciente)
            
        epoch_loss_history = np.append(epoch_loss_history, np.mean(np.array(loss_patient)))
        print('Fin epoca {}: {}'.format(epoch+1, get_tiempo()))

        # # Calculemos el loss del val:
        val_loss = get_val_loss(model, val_patients, batch_size, loss_type = loss_type)
        epoch_val_loss_history = np.append(epoch_val_loss_history, val_loss)

        if save_epochs is not None:
            if epoch//save_epochs == epoch/save_epochs and epoch>1:
                save_model(model, path2savefiles, model_name= 'model-epoch{}'.format(epoch))
                if save_plots:
                    data_dict ={
                        'epoch_loss_history': epoch_loss_history,
                        'batch_loss_history': batch_loss_history,
                        'patient_loss_history': patient_loss_history,
                        'epoch_val_loss_history': epoch_val_loss_history
                        }
                    plot(data_dict, show=plot_metrics, path_save=path2savefiles, name_plot= 'loss_epoch_{}'.format(epoch+1))

        print('Train Epoch: {}\t Train Loss: {:.6f}. Val Loss: {:.6f}'.format(
            epoch+1, epoch_loss_history[-1], epoch_val_loss_history[-1]))
        print('-----------------------------------')
    print('Fin de entrenamiento: {}'.format(get_tiempo()))
    data_dict ={
                'epoch_loss_history': epoch_loss_history,
                'batch_loss_history': batch_loss_history,
                'patient_loss_history': patient_loss_history,
                'epoch_val_loss_history': epoch_val_loss_history
                }
    save_model(model, path2savefiles, model_name='finalmodel', extension=model_extension)
    if save_plots:
        plot(data_dict, show=plot_metrics, path_save=path2savefiles)
    else:
        plot(data_dict, show=plot_metrics)
    

def checks_alright(args):
    if not args.n_epochs > 0 and not isinstance(args.n_epochs, int):
        raise ValueError("n_epochs no es entero y >0")
    if not args.batch_size > 0 and not isinstance(args.batch_size, int):
        raise ValueError("batch_size no es entero y >0")
    
    # print(args.val_split > 0 and not args.val_split < 1 and isinstance(args.val_split, float))
    if not args.val_split > 0 and not args.val_split < 1 and not isinstance(args.val_split, float):
        print('fallooooo')
        raise ValueError("val_split no es float y >0 y >1")
    
    try:
        if not os.path.exists(args.path2dataset):
            raise ValueError("La ruta al dataset, path2dataset, esta mal")
    except:
        raise ValueError("La ruta al dataset, path2dataset, esta mal")
    try:
        if not os.path.exists(args.path2savefiles):
            raise ValueError("La ruta a donde se guardaran los archivos, path2savefiles, esta mal")
    except:
        raise ValueError("La ruta a donde se guardaran los archivos, path2savefiles, esta mal")
    
    if not isinstance(args.plot_metrics, bool):
        raise ValueError("save_plots no es booleano")
    
    if not isinstance(args.plot_metrics, bool):
        raise ValueError("save_plots no es booleano")
    
    if not isinstance(args.save_epochs, int):
        if  args.save_epochs > 0 and not args.save_epochs < args.n_epochs:
            raise ValueError("save_epochs no es entero y >0 y menos que n_epochs")



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # Agregar los argumentos necesarios
    parser.add_argument('--n_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--path2dataset', type=str, default='../../manifest-1675801116903/LIDC-IDRI/')
    parser.add_argument('--path2savefiles', type=str, default='./')
    parser.add_argument('--plot_metrics', action='store_true', default = False)
    parser.add_argument('--save_plots', action='store_true', default = True)
    parser.add_argument('--save_epochs', type=int, default=None)
    parser.add_argument('--model_extension', type=str, default='.pt')
    parser.add_argument('--loss_type', type=int, default=1)
    # Obtener los argumentos proporcionados por el usuario
    args = parser.parse_args()
    checks_alright(args)

    archivo = open('./failed_patients.txt', 'r')  # Reemplaza 'nombre_archivo.txt' por el nombre de tu archivo

    failed_patients = []

    for linea in archivo:
        linea = linea.strip()  # Elimina los espacios en blanco al principio y al final de la línea
        failed_patients.append(linea)
    archivo.close()
    print('Descargando el modelo...')
    # Descargamos el modelo preentrenado:
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                       in_channels=3, out_channels=1, init_features=32, pretrained=True)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device)
    
    # Llamar a la función train con los argumentos
    train(model, n_epochs=args.n_epochs, batch_size=args.batch_size, val_split=args.val_split,
        path2dataset=args.path2dataset, path2savefiles=args.path2savefiles,
        plot_metrics=args.plot_metrics, save_plots=args.save_plots,
        save_epochs=args.save_epochs, failed_patients=failed_patients,
        model_extension=args.model_extension, loss_type=args.loss_type)
        
        
        

