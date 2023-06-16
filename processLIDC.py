import pylidc as pl
from pylidc.utils import consensus
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
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
            cmask, cbbox, _ = consensus(ann_clust, clevel=0.5,
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
    
    def scale(self):
        """Sirve para escalar los datos y que se guarden en el atributo
        self.imgs_scaled. Aunque tambien vale para comparar con y sin
        escalado las imagen y su histograma de valores."""
        imgs = np.copy(self.vol)  # +mask[:, :, i]*1000

        # Escalado:
        mini = np.min(imgs[imgs >=-2000])
        imgs=imgs-mini
        imgs[imgs <=0] = 0
        self.imgs_scaled = np.log(imgs+1)

    def imshow(self, slices=(0,), label=True, scaled=True, path2save=None):
        """gpu = True es par aindicar que el modelo ha sido entrenado con la grafica y por tanto,
        el tnsor de datos qu debe comerse es de cuda tensor"""
        print('obteniendo los datos...')
        mask = [self.mask[:, :, i] for i in slices]
        if scaled:
            images = [self.imgs_scaled[:, :, i] for i in slices]
        else:
            images = [self.vol[:, :, i] for i in slices]

        num_images = len(slices)
        rows = int(math.sqrt(num_images))+1
        cols = math.ceil(num_images / rows)

        fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
        axes = axes.flatten()
        for i, image in enumerate(images):

            axes[i].imshow(image)
            axes[i].axis('off')
            axes[i].set_title('Slice {}'.format(slices[i]))
            if label:
                contours = axes[i].contour(mask[i], levels=[0.5], colors='blue')
                axes[i].clabel(contours, inline=True, fontsize=8)
        fig.legend()
        if num_images < len(axes):
            for j in range(num_images, len(axes)):
                fig.delaxes(axes[j])
        fig.suptitle('{}'.format(self.id_patient))
        plt.tight_layout()
        plt.show()
        if path2save is not None:
            fig.set_facecolor('white')
            path = '{}/pred_grid_{}.png'.format(path2save, self.id_patient)
            fig.savefig('{}'.format(path), dpi=300, bbox_inches='tight')
            print('figura {} guardada'.format(path))

