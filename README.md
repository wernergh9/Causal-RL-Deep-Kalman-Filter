#### Deep Kalman Filters para Aprendizaje Reforzado Profundo Causal


Proyecto programado con Python 3.8.10 y Pytorch 1.12.1+cu113.

##### Instalación 
Pasos de instalación:

1. Instalar paquetes en ```requirements.txt```. En caso de usar pip con el comando ```pip install -r```, se recomienda instalar Pytorch 1.12.1+cu113 previamente. Alternativamente, los paquetes necesarios son: ```numpy==1.23.4```, ```PyYAML==6.0```, ```tensorboard==2.11.0```, ```tqdm==4.64.1```, ```termcolor==2.1.0```, ```pandas==1.5.1```, ```imageio==2.22.4```, ```imageio-ffmpeg==0.4.7```, ```opencv-python==4.6.0.66```, ```Gymnasium==0.26.3```, ```scikit-learn==1.1.3```. 
    

2. Clonar repositorio Minigrid ```https://github.com/Farama-Foundation/Minigrid```.

3. Dentro del repositorio de Minigrid, ejecutar ```python -m pip install -e .```.
```python setup install```.

##### Ejecución

Los tres principales scripts son ```train_policy.py```, ```train_dkf.py``` y ```train_policy_imaginary.py```. Todos los archivos tienen como argumento ```--cfg_path```, que requiere la ruta hacia un archivo de configuración YML para el script. Estos archivos de configuracion son especializados para cada script y se encuentran dentro de la carpeta ```/config/```. 

Los scripts generarán carpetas de experimentos donde guardarán los resultados y la configuración utilizada. Cada script generará además un archivo para utilizar con tensorboard y visualizar el tiempo real el desempeño de las métricas de cada algoritmo. Los archivos ```train_dkf.py``` y ```train_policy_imaginary.py``` generarán además videos con el desempeño del algoritmo de aprendizaje reforzado. Scripts deben ser ejecutados desde la misma carpeta donde se encuentran.

1. ```train_policy.py``` ejecuta el aprendizaje de la política experta.  El archivo de configuración utilizado es ```/config/Rainbow.yml```. Ejemplo de uso: ```python train_policy.py --cfg_path config/Rainbow.yml```.

2. ```train_dkf.py``` ejecuta el aprendizaje del Deep Kalman Filter. Su configuración requiere de un path al modelo de política experta junto al ambiente en el que fue entrenada, por lo que es necesario editar el archivo al entrenar la política experta. Ejemplo de uso: ```python train_dkf.py --cfg_path config/DKF.yml```.

3. ```train_policy.py``` ejecuta el aprendizaje de la política experta. Al igual que el script anterior, necesita un modelo entrenado para funcionar. En este caso, es necesario editar el archivo de configuración para incluir un path hacia el checkpoint de un DKF entrenado. Ejemplo de uso ```python train_policy_imaginary.py --cfg_path config/Imagine.yml```.