# MedAssist-AI-IMAGE — Clasificación Visual de Medicamentos

El objetivo es clasificar imágenes de medicamentos españoles en cinco grupos terapéuticos, combinando técnicas de Machine Learning clásico, Deep Learning y un módulo generativo para apoyar a personas mayores facilitando sus cuidados diarios.

El dataset fue construido desde cero usando la API del CIMA (https://cima.aemps.es/cima/publico/home.html), que proporciona datos estructurados e imágenes de los medicamentos registrados en España.

---

## Índice

1. [Descripción del problema](#descripción-del-problema)
2. [Dataset](#dataset)
3. [Análisis Exploratorio (EDA)](#análisis-exploratorio-eda)
4. [Machine Learning](#machine-learning)
5. [Deep Learning](#deep-learning)
6. [Módulo Generativo (GenAI)](#módulo-generativo-genai)
7. [Resultados globales](#resultados-globales)
8. [Conclusiones](#conclusiones)
9. [Estructura del repositorio](#estructura-del-repositorio)


---

## Descripción del problema

La tarea consiste en clasificar imágenes de medicamentos —fotografías de su envase (`caja`) y de la forma farmacéutica (`pastilla`)— en cinco macroclases terapéuticas:

| # | Clase |
|---|-------|
| 1 | Cardiovascular |
| 2 | Neurología y psiquiatría |
| 3 | Antiinfecciosos sistémicos |
| 4 | Respiratorio |
| 5 | Otros (agrupa las 11 macroclases restantes) |

El diseño de estas clases responde a un criterio de viabilidad: las cuatro primeras son las más frecuentes en el dataset y tienen suficiente volumen para entrenar y evaluar modelos de forma rigurosa. El resto se agrupa en `otros`, una clase intencionalmente heterogénea que permite medir si el modelo detecta cuándo un medicamento no pertenece a ninguna de las categorías principales.

**¿Por qué no una clasificación más granular?** Se evaluaron tres alternativas antes de decidir:

- **Binaria (caja vs. pastilla)**: visualmente trivial y sin valor práctico.
- **Por nombre comercial**: con una media de 2 imágenes por medicamento y más de 5.000 clases, no es viable con supervisión clásica.
- **Por macroclase terapéutica (enfoque adoptado)**: punto de equilibrio entre complejidad real del problema, volumen de datos disponible y utilidad del sistema.

---

## Dataset

### Fuente de datos

El dataset fue construido a partir de la API del CIMA, que devuelve información estructurada (JSON) de cada medicamento registrado en la AEMPS, incluyendo rutas a las imágenes asociadas.

### Proceso de construcción

1. **Lectura del JSON de medicamentos** — Se extrajeron los campos relevantes: nombre, número de registro, principio activo, clasificación ATC (niveles 1–5), forma farmacéutica, vía de administración, laboratorio titular, indicador de genérico y condición de prescripción.

2. **Indexación de imágenes** — Se localizaron todos los archivos de imagen válidos (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tif`) y se organizaron para su emparejamiento con cada medicamento.

3. **Emparejamiento imagen–medicamento** — Se utilizó el campo `nregistro` como clave de unión entre los registros del JSON y los archivos de imagen. Se normalizaron los tipos de imagen: `materialas` → `caja` y `formafarmac` → `pastilla`.

4. **Construcción de la etiqueta** — A partir del primer nivel de la clasificación ATC se derivó la variable `macroclase_15_nombre`, que agrupa los medicamentos en 15 categorías terapéuticas generales. Esta es la etiqueta principal del proyecto.

5. **Generación del CSV** — La unidad de análisis es la imagen individual. Cada fila del dataset contiene la ruta a la imagen y todos los metadatos del medicamento asociado. Se evitaron duplicados exactos dentro del mismo medicamento.

### Estadísticas del dataset

| Métrica | Valor |
|---------|-------|
| Total de imágenes | 9.998 |
| Medicamentos únicos | 5.021 |
| Imágenes de caja (envase) | 5.021 |
| Imágenes de pastilla | 4.977 |
| Imágenes con macroclase asignada | 9.998 (100 %) |
| Macroclases únicas en el dataset | 15 |

### Variables principales

Entre las variables del dataset se encuentran: `sample_id`, `image_path`, `image_type` (`caja` / `pastilla`), `nregistro`, `nombre_normalizado`, `principio_activo_normalizado`, códigos y nombres ATC de los niveles 1 a 5, `macroclase_15_nombre`, `forma_farmaceutica_simplificada`, `via_administracion`, `receta`, `generico`, y `labtitular`.

### Partición train / val / test

La partición se realizó de forma estratificada **a nivel de `nregistro`** (medicamento), no de imagen. Esto garantiza que todas las imágenes de un mismo medicamento caigan en el mismo subconjunto, evitando data leakage entre splits. La distribución es 70 % train / 15 % val / 15 % test.

---

## Análisis Exploratorio (EDA)

### Desequilibrio de clases

Con las 5 clases definidas, la distribución presenta un ratio max/min de aproximadamente **7,6x**. La clase `otros` concentra el 30 % de las imágenes y Cardiovascular, la más representada entre las cuatro principales, tiene 1.807 imágenes.

Un modelo entrenado sin corrección tiende a predecir siempre la clase mayoritaria, obteniendo una accuracy artificialmente alta con un rendimiento muy bajo en las clases pequeñas. Por este motivo, la métrica principal en todos los experimentos es el **F1-macro**, que promedia el F1 de cada clase sin ponderar por su tamaño.

### Propiedades físicas de las imágenes

El análisis de propiedades de las imagenes indicó que:

- La anchura se concentra en torno a 200 px en todas las clases. La altura presenta algo más de variabilidad pero con distribuciones muy solapadas entre categorías.
- El tamaño en KB tiene medianas similares entre clases, con presencia de outliers pero sin diferencias estructurales.
- La mayoría de las imágenes tiene aspect ratio mayor que 1 (apaisadas), tanto en `caja` como en `pastilla`.
- Las propiedades físicas básicas no son discriminativas por sí solas: el modelo tendrá que aprender de los patrones visuales del contenido.

La resolución objetivo elegida para todos los modelos fue **224×224 píxeles**, el tamaño estándar de ImageNet y el más compatible con las arquitecturas preentrenadas usadas en la fase de Deep Learning.

### Análisis visual por clase

El análisis de histogramas de color, brillo, contraste y textura mostró que las cinco clases tienen perfiles visuales **parcialmente distinguibles pero con alta variabilidad interna**.

- Las imágenes de `caja` presentan fondos blancos dominantes en todas las clases, con variaciones en el texto impreso y el color del packaging.
- Las imágenes de `pastilla` muestran mayor diversidad de color y forma según la categoría: los medicamentos cardiovasculares y neurológicos incluyen muchos comprimidos recubiertos de colores variados, mientras que los respiratorios incluyen inhaladores con formas más complejas.
- La imagen media de cada clase es relativamente borrosa, lo que indica alta variabilidad interna y descarta descriptores simples como la comparación directa de píxeles.

### Estrategia de balanceo

Se aplicó una estrategia en dos capas:

**Data augmentation offline**: para las clases con menos imágenes que la más frecuente se generaron imágenes sintéticas mediante transformaciones realistas: flip horizontal, rotación ±20°, zoom, ajuste de brillo, contraste y saturación, y ruido gaussiano. Las transformaciones se aplicaron manteniendo la proporción `caja`/`pastilla` original de cada clase. Las imágenes sintéticas se asignaron exclusivamente al conjunto de train.

**Compensación durante el entrenamiento**: `class_weight='balanced'` en los modelos de ML clásico y `WeightedRandomSampler` en los modelos de Deep Learning, para que cada clase contribuya por igual a la función de pérdida independientemente de su tamaño.

---

## Machine Learning

### Marco general

En ML clásico, la clasificación de imágenes sigue dos pasos diferenciados: (1) transformar cada imagen en un vector de números mediante extracción manual de características y (2) entrenar un clasificador sobre esos vectores. Este enfoque es interpretable, entrena rápidamente y sirve como baseline para evaluar cuánto aporta el Deep Learning.

**Nota sobre el data augmentation en ML**: aunque el pipeline estaba diseñado para usar las imágenes aumentadas offline generadas en el EDA, un problema con las rutas de archivos hizo que los modelos de ML fueran entrenados únicamente con las imágenes originales. Los resultados deben interpretarse como un baseline conservador; el rendimiento mejoraría con el augmentation correctamente integrado.

### Extracción de características

Cada imagen de 224×224 píxeles (150.528 valores brutos) se resume en un vector de aproximadamente **1.918 números** mediante tres bloques complementarios:

**Color — histograma HSV con pirámide espacial (128 features)**

Se convierte la imagen de RGB a HSV porque el espacio HSV separa el tono (H) de la iluminación (V). Dos fotos del mismo objeto con diferente luz tienen RGB distintos pero H similar. La pirámide espacial 2×2 divide la imagen en cuatro cuadrantes y calcula un histograma del canal H (32 bins) por cuadrante. Así se captura información posicional: "el azul está arriba-izquierda, el blanco abajo-derecha", algo que un histograma global perdería.

**HOG — Histogram of Oriented Gradients (1.764 features)**

HOG detecta las direcciones de los bordes de la imagen. Se calcula sobre la imagen en escala de grises redimensionada a 128×128, dividida en celdas de 16×16 píxeles. Por cada celda se obtiene un histograma de 9 orientaciones de gradiente, normalizado en bloques de 2×2 celdas (norma L2-Hys) para robustez frente a cambios locales de iluminación. Es el descriptor clásico para objetos con formas definidas: cajas rectangulares, pastillas redondas, texto impreso.

**LBP — Local Binary Patterns (26 features)**

LBP codifica la textura local comparando cada píxel con sus 24 vecinos en un círculo de radio 3. La variante "uniform" reduce los 2²⁴ patrones posibles a solo 26 (los que tienen ≤ 2 transiciones 0→1), que son los más informativos (bordes, esquinas, líneas). El resultado es un histograma normalizado de 26 dimensiones, invariante a cambios monotónicos de iluminación.

La combinación de los tres bloques permite al clasificador ver simultáneamente el color dominante del envase, las formas y bordes del objeto, y las texturas del diseño gráfico.

### Preprocesamiento

Se aplicó `StandardScaler` ajustado exclusivamente sobre el conjunto de train para llevar todas las features a media 0 y desviación estándar 1. Val y test se transformaron con los parámetros aprendidos de train. Esto es una práctica obligatoria: ajustar el scaler usando val o test constituiría una forma de data leakage.

Para los modelos sensibles a la dimensionalidad (SVM y KNN), se redujo la representación con PCA al 95 % de varianza explicada, pasando de 1.918 features a aproximadamente 100 componentes principales.

### Modelos evaluados

#### Baselines (DummyClassifier)

Se establecieron dos referentes:
- **Mayoría de clase**: predice siempre la clase más frecuente. Obtiene una accuracy igual a la proporción de la clase dominante (~50,5 %). F1-macro = 0.134.
- **Aleatorio estratificado**: predice según la distribución del train. F1-macro ≈ 0.20.

Cualquier modelo real debe superar claramente estos baselines; si no lo hace, las features no contienen señal útil.

#### Logistic Regression

Modelo lineal que aprende un conjunto de pesos para cada feature y clase, aplica la función softmax multiclase y predice la clase de mayor probabilidad. Se usó `class_weight='balanced'`, solver `lbfgs` con optimización multinomial real, y regularización L2 con C=1.0.

La regresión logística es el baseline supervisado más directo: rápido, interpretable y útil para confirmar que existe señal lineal en el espacio de features.

**Resultado en validación**: Accuracy = 0.395 | F1-macro = 0.315

Aunque la accuracy es inferior al baseline de mayoría, el F1-macro supera al baseline en más de 18 puntos. El modelo deja de predecir siempre la clase dominante y comienza a repartir predicciones entre todas las clases. La matriz de confusión muestra aciertos reales en las cinco categorías.

#### K-Nearest Neighbors (KNN)

Modelo sin entrenamiento propiamente dicho: para clasificar una imagen nueva, busca sus K vecinos más cercanos en el espacio de features (con reducción PCA previa para evitar la maldición de la dimensionalidad) y vota por mayoría ponderada por distancia.

Se usó K=10 con pesos inversamente proporcionales a la distancia y métrica euclídea. El uso de PCA (≈100 dims en lugar de 1.918) es crucial porque en alta dimensión todas las distancias tienden a ser similares, degenerando la noción de "vecino cercano".

**Resultado en validación**: Accuracy = 0.542 | F1-macro = 0.359

KNN mejora en accuracy y F1-macro gracias a que la estructura local del espacio de features contiene información útil. Sin embargo, parte de esa mejora se debe a una mayor tendencia a predecir `otros` (recall = 0.81), lo que penaliza las clases minoritarias.

#### Support Vector Machine (SVM con kernel RBF)

SVM busca el hiperplano que maximiza el margen entre clases. Con el kernel RBF proyecta los datos a un espacio de mayor dimensión donde las clases son linealmente separables, midiendo la similitud entre puntos mediante una función gaussiana. Se usaron C=10, `gamma='scale'`, `class_weight='balanced'`, con los datos reducidos por PCA.

El kernel RBF captura relaciones no lineales que la regresión logística no puede modelar, lo que lo convierte en el modelo clásico más potente para tamaños de dataset medianos.

**Resultado en validación**: Accuracy = 0.612 | F1-macro = 0.485

Es el mejor modelo clásico. La mejora frente a KNN no se debe únicamente a predecir más `otros`, sino a un rendimiento más equilibrado en Cardiovascular, Neurología y psiquiatría y Respiratorio. Persiste confusión hacia `otros` en Antiinfecciosos y Respiratorio.

#### Random Forest

Ensemble de 300 árboles de decisión entrenados con aleatoriedad: cada árbol usa un bootstrap del train y solo considera `sqrt(n_features)` features aleatorias en cada split. La predicción final es el voto de todos los árboles.

**Resultado en validación**: Accuracy = 0.551 | F1-macro = 0.273

Accuracy alta pero F1-macro bajo: el modelo concentra sus aciertos en `otros` (recall = 0.97) y prácticamente no detecta Antiinfecciosos, Neurología ni Respiratorio (recalls < 0.10). No es el mejor modelo para este problema a pesar de su alta accuracy global.

#### Gradient Boosting

Árboles secuenciales donde cada nuevo árbol corrige los errores del anterior. Se usaron 150 árboles de profundidad 4, learning rate 0.10 y subsample 0.80, con reducción PCA previa.

**Resultado en validación**: Accuracy = 0.529 | F1-macro = 0.258

Comportamiento similar a Random Forest: buena accuracy aparente pero muy sesgado hacia `otros`. No mejora al SVM.

### Optimización con GridSearchCV

Se aplicó GridSearchCV sobre el mejor modelo (SVM RBF) con una rejilla de valores para C (1, 10, 100) y gamma (`scale`, 0.01, 0.001), con validación cruzada estratificada de 5 folds. Los mejores parámetros encontrados no variaron sustancialmente respecto a los iniciales, lo que confirma que la configuración base ya estaba bien ajustada.

**SVM RBF optimizado**: Accuracy = 0.618 | F1-macro = 0.494

La mejora es marginal, lo que indica que el límite del rendimiento no depende únicamente de los hiperparámetros del modelo, sino de las features disponibles.

### Diagnóstico de overfitting

Todos los modelos presentan un gap train–val considerable. El SVM tiene el menor gap (≈0.36 de accuracy), lo que refuerza su posición como el modelo más estable. La Regresión Logística muestra el mayor gap (≈0.51). Estos valores son elevados pero coherentes con la dificultad del problema y con el hecho de que el augmentation no pudo integrarse correctamente.

### Ablation studies

Se realizaron tres estudios de ablación usando Random Forest como modelo pivote (rápido y robusto):

1. **Efecto del data augmentation**: los modelos entrenados con imágenes augmentadas tienden a generalizar mejor en clases minoritarias.
2. **Efecto del balanceo con `class_weight`**: sin balanceo, los modelos colapsan hacia `otros`. El balanceo mejora el F1-macro en todas las clases minoritarias.
3. **Contribución de cada bloque de features**: HOG es el bloque más informativo individualmente. Color aporta señal complementaria. LBP, el más compacto (26 dims), mejora marginalmente el rendimiento al combinarse con los otros dos.

### Evaluación final en test (SVM RBF optimizado)

El modelo optimizado se evaluó una sola vez en el conjunto de test:

| Métrica | Val | Test | Δ |
|---------|-----|------|---|
| Accuracy | 0.618 | ~0.61 | < 0.01 |
| F1-macro | 0.494 | ~0.48 | < 0.02 |

La consistencia entre val y test confirma que la metodología de partición y selección de modelo no introdujo sobreajuste al conjunto de validación.

**Pipeline exportado**: `scaler → pca → SVM RBF`, guardado en `output_ml/pipeline_ml_final.pkl`.

---

## Deep Learning

### Marco general

El Deep Learning aprende automáticamente qué representaciones son útiles para clasificar, sin necesidad de diseñar features a mano. La red convolucional transforma la imagen en capas de representación progresivamente más abstractas: bordes y texturas en las capas iniciales, formas y objetos en las intermedias, y combinaciones semánticas en las finales.

Se evaluaron cinco arquitecturas organizadas en dos categorías:

- **CNN desde cero (scratch)**: redes entrenadas con los pesos inicializados aleatoriamente.
- **Transfer Learning**: modelos preentrenados en ImageNet con sus capas superiores adaptadas al dominio farmacéutico.

La métrica principal sigue siendo el **F1-macro** sobre el conjunto de validación. El criterio de parada anticipada (early stopping) se aplicó sobre esta misma métrica con una paciencia de 7 épocas.

### Configuración común

- Imagen: 224×224 píxeles, normalización con media y std de ImageNet.
- Batch: 32 imágenes. Sampler ponderado (`WeightedRandomSampler` con pesos `1/sqrt(n_clase)`) para garantizar representación equilibrada de todas las clases en cada batch.
- Función de pérdida: `CrossEntropyLoss` con `label_smoothing=0.05` (en transfer learning, 0.12). Label smoothing reduce la confianza excesiva del modelo asignando una pequeña probabilidad a todas las clases, lo que mejora la calibración.
- Optimizador: `AdamW` con weight decay. Gradient clipping a norma 1.0 para estabilidad.

### CNN desde cero

#### Por qué se esperan resultados inferiores al ML en este contexto

Con menos de 15.000 imágenes originales y cinco clases sin firma visual consistente, entrenar una CNN desde cero tiene limitaciones estructurales:

- La red ve cada imagen pocas veces y no converge a representaciones robustas.
- "Cardiovascular" puede ser una caja azul, blanca, roja o verde según el laboratorio: no hay un patrón visual estable por clase.
- `otros` agrupa medicamentos heterogéneos sin ningún patrón compartido.
- El ML clásico usa HOG + LBP + Color, features diseñadas durante décadas para exactamente este tipo de imagen. La CNN debe aprender features equivalentes desde cero.

El objetivo de las redes scratch no es superar al ML, sino documentar el techo natural y motivar el uso de transfer learning.

#### MedCNN — arquitectura VGG-style

Cuatro bloques Conv-BatchNorm-ReLU-MaxPool clásicos, con un clasificador completamente conectado al final. Se entrenó con Focal Loss (γ=2) para reducir el peso de los ejemplos fáciles y dar más importancia a las clases difíciles.

**Resultado**: Accuracy val ≈ 0.482 | F1-macro ≈ 0.264

El modelo aprende patrones superficiales pero no converge a representaciones robustas. Cardiovascular obtiene F1 = 0.00 (la red no aprende a distinguirla), mientras que `otros` absorbe la mayoría de las predicciones.

Los mapas Grad-CAM revelan que el modelo activa zonas de contraste y bordes generales del envase, incluyendo la regla de escala presente en muchas imágenes, que es un elemento externo al medicamento. Esto indica que el modelo está aprendiendo correlaciones accidentales del dataset.

#### MedCNN_Res — con skip connections

La misma capacidad que MedCNN pero con conexiones residuales entre bloques. Las skip connections permiten que el gradiente fluya directamente desde la pérdida hasta las capas iniciales, resolviendo el problema del vanishing gradient. La formulación es:

```
x ──┬── Conv-BN-ReLU ── Conv-BN ──┬── ReLU → salida
    └────── (shortcut 1×1) ────────┘
```

**Resultado**: Accuracy val ≈ 0.470 | F1-macro ≈ 0.376

Las skip connections mejoran el F1-macro en +0.11 respecto a la versión VGG. Cardiovascular pasa de F1=0.00 a 0.221, Neurología y psiquiatría alcanza 0.459 y Antiinfecciosos 0.329. Sin embargo, la curva de entrenamiento muestra un gap train–val de 0.252 (accuracy train ≈ 0.74 vs. val ≈ 0.47), indicando sobreajuste.

Los mapas Grad-CAM muestran un foco ligeramente más coherente: el modelo empieza a mirar el cuerpo del objeto, aunque sigue contaminado por la regla de escala y zonas de contraste no informativas.

### Transfer Learning

Transfer Learning consiste en partir de un modelo ya entrenado en un corpus grande (ImageNet, 1.2M imágenes, 1.000 clases) y adaptarlo a la tarea objetivo. El modelo ya sabe detectar bordes, texturas, formas, colores y objetos; solo tiene que aprender qué combinaciones de esas representaciones son relevantes para clasificar medicamentos.

#### Estrategia de fine-tuning en dos fases

**Fase 1 — backbone congelado (5 épocas)**

Se sustituye la cabeza clasificadora original por una nueva: `Dropout(0.5) → Linear(in_features, 128) → ReLU → Dropout(0.3) → Linear(128, 5)`. El backbone queda congelado y solo se entrena la cabeza nueva con learning rate alto (4×lr_base) y OneCycleLR.

Esta fase es necesaria porque la cabeza empieza con pesos aleatorios. Si se desbloqueara todo el modelo desde el inicio, los gradientes inestables de la cabeza dañarían las representaciones preentrenadas del backbone.

**Fase 2 — fine-tuning parcial de los bloques finales**

Se descongelan solo los últimos bloques del backbone y se entrena con learning rate diferencial: el backbone con 0.05×lr_base (muy conservador) y la cabeza con lr_base completo. El scheduler es CosineAnnealingLR.

La decisión de descongelar solo los bloques finales (en lugar de todo el modelo) está fundamentada: las capas iniciales detectan bordes, texturas y contrastes que son útiles para cualquier imagen, incluyendo medicamentos. Las capas finales contienen representaciones más específicas del dominio y son las que necesitan adaptarse. Con un dataset pequeño, desbloquear todo el backbone aumenta el riesgo de sobreajuste porque el modelo tiene demasiados grados de libertad para ajustarse a las correlaciones accidentales del train.

Los bloques descongelados en cada arquitectura fueron:
- **EfficientNet-B0**: `features.6`, `features.7`, `features.8`, `classifier` (~1.5M parámetros).
- **ResNet50**: `layer3`, `layer4`, `fc` (~14M parámetros, 93.9 % del total).
- **MobileNetV3-Large**: `features.14`, `features.15`, `features.16`, `classifier` (~2M parámetros).

#### EfficientNet-B0

Arquitectura con 5.3M parámetros y 0.4 GFLOPS, diseñada mediante neural architecture search para maximizar el ratio rendimiento/coste. Su escalado compuesto (anchura, profundidad y resolución) permite obtener un rendimiento alto con muy pocos parámetros.

**Resultado**: Accuracy val = 0.507 | F1-macro = 0.434

Mejora clara sobre las CNN scratch. Gap train–val ≈ 0.022, lo que indica ausencia de sobreajuste grave. Por clase: Cardiovascular 0.318, Neurología y psiquiatría 0.451, Antiinfecciosos 0.348, Respiratorio 0.419, Otros 0.632.

Los mapas Grad-CAM muestran que EfficientNet empieza a mirar zonas de texto y color del envase (más razonable que las scratch), aunque las predicciones siguen siendo incorrectas en varios casos. El modelo localiza el medicamento, pero localizar no equivale a entender la categoría terapéutica.

#### ResNet50

Arquitectura con 25.6M parámetros y bloques residuales profundos (50 capas). Su mayor capacidad representacional permite aprender combinaciones más complejas de textura, forma, color y estructura del envase.

**Resultado**: Accuracy val = 0.551 | F1-macro = 0.461

Es el **mejor modelo de Deep Learning** del proyecto. Gap train–val ≈ 0.007, el más bajo de todos los modelos evaluados. Por clase: Cardiovascular 0.299, Neurología y psiquiatría 0.459, Antiinfecciosos 0.411, Respiratorio 0.453, Otros 0.682.

ResNet50 combina el mejor F1-macro con la generalización más estable. La curva de pérdida baja de forma progresiva tanto en train como en val, sin el patrón de sobreajuste que aparecía en las redes scratch.

A pesar de desbloquear el 93.9 % de sus parámetros en la fase 2, no presenta sobreajuste grave gracias a la combinación de learning rate muy bajo en el backbone, dropout, weight decay, label smoothing y data augmentation online.

Los mapas Grad-CAM muestran un foco visual más razonable: el modelo activa zonas frontales del envase, cerca del texto y las franjas de color. Sin embargo, sigue sufriendo de shortcut learning en algunos casos, activándose en la regla de escala en lugar de en el texto del medicamento.

#### MobileNetV3-Large

Arquitectura de 5.5M parámetros y 0.22 GFLOPS, diseñada para despliegue eficiente en dispositivos móviles o APIs con restricciones de latencia.

**Resultado**: Accuracy val = 0.511 | F1-macro = 0.405

Mejor que las CNN scratch, peor que EfficientNet-B0 y ResNet50. Gap train–val ≈ 0.037. Su principal ventaja es la eficiencia de inferencia; para una posible API en producción, MobileNetV3 ofrece el mejor balance velocidad/precisión.

#### Variante con fine-tuning completo (experimento adicional)

Se realizaron pruebas adicionales desbloqueando todo el backbone (no incluidas en el notebook por coste computacional). Los resultados muestran que el fine-tuning completo puede aumentar el F1-macro de ResNet50 hasta ≈ 0.528, pero con un gap train–val que sube hasta ≈ 0.174, indicando sobreajuste. El enfoque conservador del notebook principal (desbloqueo parcial) es metodológicamente más sólido para el tamaño de dataset disponible.

### Comparativa completa ML vs DL

| Modelo | Tipo | F1-macro (val) | Accuracy (val) |
|--------|------|---------------|----------------|
| ResNet50 (TL) | Transfer Learning | **0.461** | **0.551** |
| EfficientNet-B0 (TL) | Transfer Learning | 0.434 | 0.507 |
| MobileNetV3-L (TL) | Transfer Learning | 0.405 | 0.511 |
| SVM RBF (ML, optimizado) | ML Clásico | 0.494 | 0.618 |
| MedCNN_Res (scratch) | CNN scratch | 0.376 | 0.470 |
| MedCNN (scratch) | CNN scratch | 0.264 | 0.482 |

El SVM supera a los modelos de Deep Learning en F1-macro. Esto no es sorprendente dado el contexto: el dataset es pequeño (< 10K imágenes originales), las clases no tienen una firma visual consistente, y el ML clásico usa features diseñadas para este tipo de imagen. El Deep Learning necesita más datos y más épocas de entrenamiento del que el dataset puede proporcionar.

Sin embargo, el F1-macro no es la única dimensión relevante. ResNet50 muestra una generalización más estable (gap train–val de 0.007 frente a 0.36 del SVM), lo que sugiere que con más datos el modelo de DL podría superar al ML clásico.

### Evaluación final en test (ResNet50)

| Métrica | Val | Test |
|---------|-----|------|
| Accuracy | 0.551 | ~0.55 |
| F1-macro | 0.461 | ~0.45 |

La consistencia entre val y test confirma una generalización adecuada. El modelo exportado incluye los pesos del checkpoint con mejor F1-macro en validación, junto con los metadatos de arquitectura, clases y preprocesamiento.

---

## Módulo Generativo (GenAI)

El módulo generativo convierte el resultado del clasificador en una experiencia accesible para personas mayores. A partir de la imagen de un medicamento, el sistema genera:

1. **Image Captioning**: una descripción visual objetiva de la imagen en lenguaje natural, usando modelos de la familia BLIP (`blip-image-captioning-large` como opción principal, con `vit-gpt2` como fallback para CPU).

2. **Descripción terapéutica con LLM**: el caption visual y la clase predicha por el clasificador se combinan en un prompt y se envían a Llama-3.1-8b-instant (API de Groq, gratuita) para generar una explicación en español simple, sin jerga médica, adaptada a personas mayores. El prompt incluye instrucciones para que el mensaje sea breve, claro y oriente sobre el uso general del medicamento.

3. **Generación de icono visual (Image-to-Image)**: si hay GPU disponible, se usa Stable Diffusion img2img para transformar la foto del medicamento en un icono representativo de su categoría terapéutica (corazón para Cardiovascular, cerebro para Neurología, etc.). En entornos sin GPU, se generan iconos programáticos con PIL.

Este módulo funciona de forma encadenada: el clasificador del notebook de DL proporciona la clase, BLIP genera la descripción visual, el LLM produce la explicación en lenguaje natural, y Stable Diffusion genera el icono. Si el clasificador no está disponible, el sistema opera en modo demo con predicciones simuladas.

---

## Resultados globales

### Resumen por modelo

| Modelo | F1-macro val | Accuracy val | Observaciones |
|--------|-------------|--------------|---------------|
| SVM RBF optimizado | 0.494 | 0.618 | Mejor F1-macro global |
| ResNet50 (TL) | 0.461 | 0.551 | Mejor generalización DL |
| EfficientNet-B0 (TL) | 0.434 | 0.507 | Mejor ratio rendimiento/parámetros |
| MobileNetV3-L (TL) | 0.405 | 0.511 | Mejor para deployment en producción |
| KNN | 0.359 | 0.542 | Segundo mejor ML |
| MedCNN_Res (scratch) | 0.376 | 0.470 | Mejor CNN sin preentrenamiento |
| Logistic Regression | 0.315 | 0.395 | Baseline supervisado |
| MedCNN (scratch) | 0.264 | 0.482 | Peor modelo real |
| Baseline mayoría | 0.134 | 0.505 | Cota inferior |

### Dificultad por clase (F1 del mejor modelo DL — ResNet50)

| Clase | F1 (val) |
|-------|----------|
| Otros | 0.682 |
| Neurología y psiquiatría | 0.459 |
| Respiratorio | 0.453 |
| Antiinfecciosos sistémicos | 0.411 |
| Cardiovascular | 0.299 |

Cardiovascular es la clase más difícil pese a ser la más frecuente, porque los medicamentos cardiovasculares tienen packaging muy heterogéneo entre laboratorios. `Otros` es la más fácil de detectar porque el modelo puede identificar "nada de lo anterior", aunque a costa de confundir allí muchos medicamentos de las otras clases.

---

## Conclusiones

**Sobre el problema**

La clasificación de medicamentos por imagen en macroclases terapéuticas es un problema con señal real pero limitada. Las categorías terapéuticas no corresponden a categorías visuales: un medicamento cardiovascular puede tener exactamente el mismo aspecto externo que un antibiótico si ambos son producidos por el mismo laboratorio o tienen envases similares. La señal más discriminativa suele estar en el texto del envase (nombre del medicamento, principio activo, dosis), no en el color ni en la forma del packaging.

**Sobre Machine Learning clásico**

El SVM con kernel RBF y features HOG+Color+LBP es el mejor modelo del proyecto en términos de F1-macro. La combinación de features manuales bien diseñadas, reducción de dimensionalidad con PCA y un clasificador no lineal resulta más eficiente que las CNN scratch para este dataset. Los ablation studies confirman que HOG es el bloque más informativo, que el balanceo de clases es imprescindible y que el data augmentation mejora la generalización en clases minoritarias.

**Sobre Deep Learning**

Transfer Learning es claramente superior a entrenar desde cero. Las CNN scratch no convergen a representaciones robustas con menos de 15.000 imágenes y clases sin firma visual estable. Los modelos preentrenados en ImageNet aportan features visuales ricas que compensan el tamaño reducido del dataset.

ResNet50 es el mejor modelo de DL: mayor capacidad representacional, generalización estable (gap train–val de 0.007) y F1-macro equilibrado entre clases. La estrategia de fine-tuning parcial (solo últimos bloques, learning rate diferencial) resulta más defensible metodológicamente que desbloquear todo el backbone.

La interpretabilidad de los modelos (Grad-CAM) revela un problema compartido: los modelos activan con demasiada frecuencia la regla de escala presente en muchas imágenes, que es un elemento externo al medicamento. Este shortcut learning limita la robustez del sistema en escenarios reales.

**Sobre la comparativa ML vs. DL**

El SVM supera a los modelos de Deep Learning en F1-macro, pero con un gap train–val mucho mayor. Con más datos y augmentation correctamente integrado, los modelos de DL probablemente superarían al SVM, porque la capacidad de aprender representaciones automáticamente escala mejor con el volumen de datos.

**Sobre el siguiente paso**

El límite actual del sistema no es la arquitectura del modelo, sino la naturaleza de las features disponibles. El siguiente salto de calidad no debería consistir únicamente en usar una CNN más grande, sino en construir un sistema multimodal que combine la información visual de la imagen con el texto del envase (mediante OCR o modelos visión-lenguaje como BLIP-2 o PaliGemma) y con los metadatos farmacológicos disponibles en el CSV.

---

## Estructura del repositorio

```
.
├── EDA_bueno.ipynb             # Construcción del dataset, EDA y data augmentation
├── ML_bueno.ipynb              # Pipeline ML: extracción de features y modelos clásicos
├── DL_final.ipynb              # Pipeline DL: CNN scratch y transfer learning
├── GenAI.ipynb                 # Módulo generativo: captioning y generación de iconos
│
├── output/
│   ├── dataset.csv             # Dataset original (una fila = una imagen)
│   ├── dataset_split.csv       # Dataset con columna split (train/val/test)
│   └── dataset_balanced.csv    # Dataset con imágenes augmentadas en train
│
├── output_ml/
│   ├── pipeline_ml_final.pkl   # Pipeline completo: scaler → pca → SVM
│   ├── label_encoder.pkl       # LabelEncoder de las 5 clases
│   ├── scaler.pkl              # StandardScaler ajustado sobre train
│   └── comparativa_modelos.csv # Métricas de todos los modelos ML
│
├── output_dl/
│   ├── best_model_resnet50.pth # Pesos del mejor modelo DL
│   ├── label_encoder_dl.pkl    # LabelEncoder (mismas clases)
│   ├── model_metadata.json     # Arquitectura, clases, métricas y preprocesamiento
│   └── comparativa_dl.csv      # Métricas de todos los modelos DL
│
├── output_gen/
│   └── ...                     # Captions e iconos generados por el módulo GenAI
│
├── feature_cache/
│   └── feats_v3_*.pkl          # Caché de features extraídas (evita recalcular)
│
└── data_augmented/
    └── train/                  # Imágenes sintéticas generadas en el EDA
```


