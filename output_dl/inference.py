#!/usr/bin/env python3
"""
inference.py — MedAssist-AI: Inferencia del modelo DL para la API.

Uso:
    from inference import MedAssistPredictor
    predictor = MedAssistPredictor('output_dl')
    result = predictor.predict('ruta/imagen.jpg')
    print(result)
    # {'class': 'Cardiovascular', 'confidence': 0.92, 'probabilities': {...}}
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import joblib


class MedAssistPredictor:
    """Clase de inferencia lista para producción."""

    def __init__(self, model_dir: str = 'output_dl'):
        model_dir = Path(model_dir)

        # Cargar metadatos
        with open(model_dir / 'model_metadata.json') as f:
            self.meta = json.load(f)

        self.classes = self.meta['classes']
        self.img_size = self.meta['img_size']

        # Reconstruir modelo
        self.model = self._build_model(self.meta['arch'], len(self.classes))
        self.model.load_state_dict(
            torch.load(model_dir / f'best_model_{self.meta["arch"]}.pth',
                       map_location='cpu')
        )
        self.model.eval()

        # Transformaciones
        self.transform = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=self.meta['imagenet_mean'], std=self.meta['imagenet_std']),
        ])

        print(f'MedAssistPredictor cargado: {self.meta["arch"]} | '
              f'Val F1={self.meta["val_f1_macro"]:.3f}')

    def _build_model(self, arch, num_classes):
        if arch == 'resnet50':
            from torchvision.models import resnet50, ResNet50_Weights
            m = resnet50(weights=None)
            m.fc = nn.Sequential(
                nn.Dropout(0.3), nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_classes)
            )
        elif arch == 'efficientnet_b0':
            from torchvision.models import efficientnet_b0
            m = efficientnet_b0(weights=None)
            in_f = m.classifier[1].in_features
            m.classifier = nn.Sequential(
                nn.Dropout(0.3), nn.Linear(in_f, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_classes)
            )
        else:
            from torchvision.models import mobilenet_v3_large
            m = mobilenet_v3_large(weights=None)
            in_f = m.classifier[0].in_features
            m.classifier = nn.Sequential(
                nn.Dropout(0.3), nn.Linear(in_f, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_classes)
            )
        return m

    @torch.no_grad()
    def predict(self, image_path: str) -> dict:
        """
        Clasifica una imagen de medicamento.

        Returns dict con:
            - class: nombre de la macroclase predicha
            - confidence: probabilidad de la clase predicha (0-1)
            - probabilities: dict con probabilidades de todas las clases
        """
        img = Image.open(image_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0)

        logits = self.model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()

        pred_idx   = probs.argmax()
        pred_class = self.classes[pred_idx]
        confidence = float(probs[pred_idx])

        return {
            'class':         pred_class,
            'confidence':    round(confidence, 4),
            'probabilities': {cls: round(float(p), 4) 
                              for cls, p in zip(self.classes, probs)},
        }


if __name__ == '__main__':
    import sys
    predictor = MedAssistPredictor()
    for img_path in sys.argv[1:]:
        result = predictor.predict(img_path)
        print(f"
{img_path}:")
        print(f"  Clase:      {result['class']}")
        print(f"  Confianza:  {result['confidence']:.1%}")
        print("  Probs:")
        for cls, prob in sorted(result['probabilities'].items(), key=lambda x: -x[1]):
            bar = '█' * int(prob * 20)
            print(f"    {cls:<35} {prob:.1%}  {bar}")
