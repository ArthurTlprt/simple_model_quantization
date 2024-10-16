# Simple model quantization

Dans ce repo, nous testons sur un modèle simple les différentes méthodes de quantification de modèle, via ses différentes implémentations.
Le modèle doit classifier des images du dataset FashionMNIST.

Nota Bene:
- Faire attention au device sur lequel est executé le modèle + les données. (cuda pour l'entrainement, et cpu pour l'évaluation)
- Veiller à ce que les paramètres du model quantifié préparé soient bien ceux donnés à l'optimizer, et non ceux du modèle de base.
- Des fonctions comme nn.Softmax ne sont pas implémentées dans le eager mode, alors que dans FX Graph oui.
- Désactivation du EMA: impact sur les poids
- Désactivation du AMP: idem

Ressource FP8:
- FP8 FORMATS FOR DEEP LEARNING: https://arxiv.org/pdf/2209.05433
- EFFICIENT POST-TRAINING QUANTIZATION WITH FP8 FORMATS: https://arxiv.org/pdf/2309.14592
- FP8 Quantization: The Power of the Exponent: https://arxiv.org/pdf/2208.09225
- https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html
- https://github.com/pytorch/FBGEMM/blob/e5d0c9448774e6bc577e7f210ecbec56b7a69f10/fbgemm_gpu/experimental/gemm/triton_gemm/fp8_gemm.py
- https://github.com/Qualcomm-AI-research/FP8-quantization/tree/main
- Doc Intel sur l'utilisation du FP8: https://github.com/intel/intel-extension-for-pytorch/tree/eda7a7c42df6f9a64e0de9c2b69304ee02f2c32a/intel_extension_for_pytorch/quantization/fp8



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

## Intel Pytorch Extension

### 5.

[Notebook](intel_fp8_training.ipynb) montrant un apprentissage d'un modèle en FP8.

### 6.

[Script](intel_convert_to_fp8.py) convertissant un modèle FP32 en DP8. En réalité, il semblerait que seule les couches denses puissent être converties pour le moment. 
La documentation indique "Use prepare_fp8 to convert modules to FP8 modules (e.g, convert nn.Linear to FP8Linear) in the model".

```text
Classifier(
  (model): Sequential(
    (0): Conv2d(1, 8, kernel_size=(3, 3), stride=(2, 2))
    (1): ReLU()
    (2): Conv2d(8, 64, kernel_size=(3, 3), stride=(2, 2))
    (3): ReLU()
    (4): AvgPool2d(kernel_size=3, stride=2, padding=0)
    (5): Flatten(start_dim=1, end_dim=-1)
    (6): Dropout(p=0.5, inplace=False)
    (7): FP8Linear(in_features=256, out_features=128, bias=True)
    (8): Dropout(p=0.5, inplace=False)
    (9): FP8Linear(in_features=128, out_features=10, bias=True)
    (10): Softmax(dim=None)
  )
)
```

### 7. 

Apprentissage bis en FP8: [script](intel_fp8_training_bis.py)

```bash
RuntimeError: Running FP8 on not supported platform.
```
## ONNX

## Nvidia TensorRT

Etc...