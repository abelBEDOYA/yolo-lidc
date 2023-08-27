import pylidc as pl
from pylidc.utils import consensus
import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
import torch
import time
import plotly.graph_objects as go
import numpy as np
from mayavi import mlab
import cv2
import math

class Patient():
    def __init__(self, id_patient):
        self.id_patient = id_patient
        self.scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == id_patient).first()
        self.vol = self.scan.to_volume(verbose=False)
        self.mask = self.get_mask()
        self.imgs_scaled = np.array([])

    def get_mask(self, print_count=False):
        mask = np.zeros_like(self.vol)
        nod_count = 0
        for ann_clust in self.scan.cluster_annotations():
            # print('hola')
            nod_count +=1
            cmask, cbbox, _ = consensus(ann_clust, clevel=0.1,
                                        pad=[(20, 20), (20, 20), (0, 0)])
            mask[cbbox] += cmask
        if print_count is True:
                print(nod_count)
        return mask

    def plot_mask(self):
        # Generar una matriz binaria aleatoria de 10x10x10
        data = self.mask
        x, y, z = np.where(data == 1)
        fig = self.__plot_3d(x,y,z,n=0)
        x_min, x_max = 0, self.mask.shape[1]
        y_min, y_max = 0, self.mask.shape[0]
        z_min, z_max = 0, self.mask.shape[2]
        fig.update_layout(scene=dict(xaxis=dict(range=[x_min, x_max]),
                                    yaxis=dict(range=[y_min, y_max]),
                                    zaxis=dict(range=[z_min, z_max])),
                        title={'text': "mask nodulos"})
        # fig.title('mascara tumores')
        fig.show()

    def __plot_3d(self, x, y, z, n=None):
        """
        Plotea en 3d (scater plot) a partir de tres arrays: x, y, z
        Siendo estos las coordenadas de cada punto
        """
        colors = ['rgba(0, 0, 255, 1)' for i in range(len(z))]
        if not n is None:
            colors[-n:] = ['rgba(255, 0, 0, 1)' for i in range(n)]
        # Crea el gráfico de dispersión 3D
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=1,
                opacity=0.1,
                # colorscale='Viridis',  # Escala de color
                # colorbar=dict(title='Valor de z'),  # Leyenda del color
                color=colors
            )
        )])

        # Configura el diseño del gráfico
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='Eje x (mm)'),
                yaxis=dict(title='Eje y (mm)'),
                zaxis=dict(title='Eje z (mm)'),
            )
        )
        return fig 


    def reconstruct_body(self, body=True, nodulos=False):
        # vol  = np.transpose( vol , [2,0,1]).astype(np.float32) # each row will be a slice

        points = {'x': np.array([]), 'y': np.array([]), 'z': np.array([])}
        if body is True:
            prob_true = 0.3
            for i in range(self.vol.shape[2]):
                # print(np.max(vol[:, :, i]))
                thresh_value = 600
                ret, thresh = cv2.threshold(
                    self.vol[:, :, i], thresh_value, 1, cv2.THRESH_BINARY)
                x, y = np.where(thresh == 1)
                arr_bool = np.random.choice([True, False], size=len(x), p=[
                                            prob_true, 1-prob_true])
                z = np.ones(x.shape)*i
                x, y, z = x[arr_bool], y[arr_bool], z[arr_bool]
                points['x'] = np.append(points['x'], x)
                points['y'] = np.append(points['y'], y)
                points['z'] = np.append(points['z'], z)

        x, y, z = points['x'], points['y'], points['z']
        n = None
        if nodulos == True:
            
            x_m, y_m, z_m = np.where(self.mask == 1)
            n = len(x_m)
            x = np.append(x, x_m)
            y = np.append(y, y_m)
            z = np.append(z, z_m)

        x, y, z = x*self.scan.pixel_spacing, y*self.scan.pixel_spacing, z*self.scan.slice_spacing
        print(len(points['x']))
        fig = self.__plot_3d(x, y, z, n=n)
        fig.show()

    def get_all_nodules(self, plot = False):
        """
        Aqui se obtienen las anoticaion haciendo la query con un join del scan con 
        annotation, a veces hay mas annotaciones que con scan simplemente. Aunque suele 
        ser nodulos repetidos, etiqeutados ligereamente distintos. 
        Esto es solo para informacion.
        """
        anns = pl.query(pl.Annotation).join(pl.Scan).filter(pl.Scan.patient_id == self.id_patient)
        print(f'Paciente: {self.id_patient}')
        print('___________________________________')
        for ann in anns:
            print(f'Paciente del nodulo: {ann.scan.patient_id}')
            print('Primera slice con el nodulo',ann.contour_slice_indices[0])
            print(f'num. slices nodulo: {ann.boolean_mask().shape[-1]}')
            # # Visualizacion interacctiva con el contour tambien:
            if plot == True:
                ann.visualize_in_scan()
            print('-----------')
        print('___________________________________')
    
    def scale(self, slices= (0, ), plot = False, with_mask = False):
        """Sirve para escalar los datos y que se guarden en el atributo
        self.imgs_scaled. Aunque tambien vale para comparar con y sin
        escalado las imagen y su histograma de valores."""
        imgs = np.copy(self.vol)  # +mask[:, :, i]*1000
        ## Histograma antes:
        imgs1 = imgs[:,:, list(slices)]
        h_antes = imgs1[:, :, 0].reshape(-1)
        
        # Escalado:
        mini = np.min(imgs[imgs >=-2000])
        imgs=imgs-mini
        imgs[imgs <=0] = 0
        self.imgs_scaled = np.log(imgs+1)
        if with_mask is True and plot == True:
            imgs = np.log(imgs+1) + self.mask*3
        else:
            imgs = np.log(imgs+1)
        
        # Histograma despues:
        imgs2 = imgs[:,:, list(slices)]
        h_despues = imgs2[:, :, 0].reshape(-1)
        
        if plot == True:
            plt.hist(h_antes)
            plt.title('sin escalar')
            plt.yscale('log')
            plt.show()
                
            plt.hist(h_despues)
            plt.title('escalado')
            plt.yscale('log')
            plt.show()
            
            
            for i in range(len(slices)):
                img1 = imgs1[:, :, i]  # +mask[:, :, i]*1000
                # ## Imagen:
                plt.imshow(img1, cmap = 'gray')
                plt.title('sin escalar')
                plt.title(f'slice: {i+slices[0]}')
                plt.show()
                img2 = imgs2[:, :, i]
                plt.imshow(img2, cmap = 'gray')
                plt.title('escalado')
                plt.title(f'slice: {slices[0]}')
                plt.show()

    def predict(self, model, slices=(0,3), scaled=True, gpu = True):
        images, mask = self.get_tensors(scaled=scaled)
        if isinstance(slices[0], str):
            slices = (0,-1)
        if torch.cuda.is_available() and gpu:
            device = torch.device('cuda')
            images, mask = images[slices[0]:slices[-1]+1,:,:].to(device), mask[slices[0]:slices[-1],:,:].to(device)
            pred = model(images)
        else:
            device = torch.device('cpu')
            images, mask = images[slices[0]:slices[-1]+1,:,:].to(device), mask[slices[0]:slices[-1],:,:].to(device)
            pred = model(images)
        if gpu:
            pred = pred.cpu().detach().numpy()
        else:
            pred = pred.detach().numpy()
        return pred

    def imshow(self, slices=(0,), label=True, scaled=True, model = None, threshold = 0.5, path2save=None, gpu = True):
        """gpu = True es par aindicar que el modelo ha sido entrenado con la grafica y por tanto,
        el tnsor de datos qu debe comerse es de cuda tensor"""
        print('obteniendo los datos...')
        images, mask = self.get_tensors(scaled=True)
        images = images.detach().numpy()
        mask = mask.detach().numpy()
        mask = [mask[i] for i in slices]
        
        if scaled:
            images = [images[i] for i in slices]
        else:
            images = [images[i] for i in slices]

        num_images = len(slices)
        
        # print(self.mask.shape)
        if model is not None:
            print('realizando inferencia...')
            legend_labels_pred = ['Predicción']
            pred = self.predict(model, slices=slices, scaled=True, gpu=gpu)
            print(pred.shape)
        legend_labels_label = ['Etiqueta']
        fig, axs = plt.subplots(1, num_images, figsize=(12, 4))
        for i in range(num_images):
            imagen = images[i][0,:,:]  # Función para obtener la imagen según el índice
            
            axs[i].imshow(imagen, cmap='gray')
            axs[i].set_title(f'Imagen {slices[i]}')
            # Dibujar contorno
            if label:
                axs[i].contour(mask[i][0], colors='blue', levels=[0.5])  # Ajusta el nivel de contorno según tus necesidades
            # print(pred[i][0].shape)
            if model is not None and np.any(pred[i,0]>threshold):
                axs[i].contour(pred[i,0], colors='red', levels=[threshold])  # Ajusta el nivel de contorno según tus necesidades
        
        plt.show()

    def get_tensors(self, scaled = True, channels_z = True):
        if scaled is False:
            vol = np.transpose(self.vol, [2, 0, 1]).astype(
            np.float32)  # each row will be a slice
            mask = np.transpose(self.mask, [2, 0, 1]).astype(
                np.float32)  # each row will be a slice
            t_vol = torch.from_numpy(vol)
            t_mask = torch.from_numpy(mask)

            # avg = torch.nn.AvgPool2d(2)
            images = t_vol
            masks = t_mask
            shape = images.shape
            # print(shape)
            images = images.view(shape[0], 1, shape[1], shape[2])
            images = images.repeat((1, 3, 1, 1))
            return images, masks
        else:
            vol = np.transpose(self.imgs_scaled, [2, 0, 1]).astype(
            np.float32)  # each row will be a slice
            mask = np.transpose(self.mask, [2, 0, 1]).astype(
                np.float32)  # each row will be a slice

            t_vol = torch.from_numpy(vol)
            t_mask = torch.from_numpy(mask)

            
            # avg = torch.nn.AvgPool2d(2)
            images = t_vol
            masks = t_mask
            shape = images.shape
            
            # Crear un nuevo tensor de shape [261, 1, 2, 512, 512] con todos los valores en cero
            transformed_masks = torch.zeros(masks.shape[0], 2, 512, 512)

            # Asignar los valores originales en la primera parte del eje 2
            transformed_masks[:, 0, :, :] = masks

            # Calcular el complemento a la unidad de los valores originales
            complement_tensor = 1 - masks

            # Asignar los valores complementarios en la segunda parte del eje 2
            transformed_masks[:, 1, :, :] = complement_tensor
            
            # print(images.shape)
            if channels_z:
                # start = time.time()
                # images_new = self.optimize_process(images)
                start1 = time.time()
                # print('funcion', start1-start)
                images_new = torch.empty((shape[0], 3, 512, 512), dtype=images.dtype)
                # print(shape[0])
                for i in range(shape[0]):
                    if i == 0:
                        images_new[i, 0, :, :] = images[i, :, :]
                    else:
                        # print('estoy aqui')
                        images_new[i, 0, :, :] = images[i - 1, :, :]
                    # print('cuidao', i)
                    images_new[i, 1, :, :] = images[i, :, :]
                    if i == shape[0]-1:
                        # print('estoy dentro')
                        images_new[i, 2, :, :] = images[i, :, :]
                    else:
                        images_new[i, 2, :, :] = images[i + 1, :, :]
                # print('a mano', time.time()-start1)
                return images_new, transformed_masks
            else:
                images = images.reshape(shape[0], 1, shape[1], shape[2])
                images = images.repeat((1,3,1, 1))
                return images, transformed_masks
    def optimize_process(self, images):
        shape = images.shape
    
        images_new = torch.empty((images.shape[0], 3, 512, 512), dtype=images.dtype)
        images_new[:, 0, :, :] = torch.roll(images[:, :, :], shifts=1, dims=0)
        images_new[:, 1, :, :] = images[:, :, :]
        images_new[:, 2, :, :] = torch.roll(images[:, :, :], shifts=-1, dims=0)
        images_new[0,0,:,:] = images[0, :, :]
        images_new[-1,2,:,:] = images[-1, :, :]
        
        return images_new
            
            
            
        
        
