# Simple model quantization

Dans ce repo, nous testons sur un modèle simple les différentes méthodes de quantification de modèle, via ses différentes implémentations.
Le modèle doit classifier des images du dataset FashionMNIST.

Nota Bene:
- Faire attention au device sur lequel est executé le modèle + les données. (cuda pour l'entrainement, et cpu pour l'évaluation)
- Veiller à ce que les paramètres du model quantifié préparé soient bien ceux donnés à l'optimizer, et non ceux du modèle de base.
- Des fonctions comme nn.Softmax ne sont pas implémentées dans le eager mode, alors que dans FX Graph oui.

Ressource FP8:
- FP8 FORMATS FOR DEEP LEARNING: https://arxiv.org/pdf/2209.05433
- EFFICIENT POST-TRAINING QUANTIZATION WITH FP8 FORMATS: https://arxiv.org/pdf/2309.14592
- FP8 Quantization: The Power of the Exponent: https://arxiv.org/pdf/2208.09225
- https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html
- https://github.com/pytorch/FBGEMM/blob/e5d0c9448774e6bc577e7f210ecbec56b7a69f10/fbgemm_gpu/experimental/gemm/triton_gemm/fp8_gemm.py
- https://github.com/Qualcomm-AI-research/FP8-quantization/tree/main



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

## Torch FX Graph implementation

### 4. Training aware quantization

[Notebook ici](fx_qat.ipynb)

## ONNX

## Nvidia TensorRT

Etc...