# Simple model quantization

Dans ce repo, nous testons sur un modèle simple les différentes méthodes de quantification de modèle, via ses différentes implémentations.
Le modèle doit classifier des images du dataset FashionMNIST.

### Le modèle simple:

Voici le [modèle simple](classifier_training.py) sur lequel on vient appliquer les différentes méthodes de quantification.

```python
self.model = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 8*8, 3, stride=2),
            nn.ReLU(),
            nn.AvgPool2d(3, 2),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(256, 128), 
            nn.Dropout(),
            nn.Linear(128, 10), 
            nn.Softmax()
        )
```


## Torch Eager Mode implementation

### 1. Post training static quantization

[Notebook ici](eager_mode_static_quant.ipynb)

### 2. Post training dynamic quantization

[Notebook ici](eager_mode_dynamic_quant.ipynb)

### 3. Training aware quantization

[Notebook ici](eager_mode_qat.ipynb)
Le modèle ne converge pas pour le moment avec ce mode.


Todo

## Torch FX Graph implementation

## ONNX

## Nvidia TensorRT

Etc...