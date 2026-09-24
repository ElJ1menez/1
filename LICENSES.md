# Licencias y uso comercial

Criterio del proyecto: **solo se usan componentes open source cuya licencia permite el
uso comercial**, tanto del código como de los **pesos de los modelos** de IA (que suelen
tener una licencia distinta a la del código, y es donde está la trampa habitual).

## AI Motion Tracker

| Componente | Para qué | Código | Pesos / modelos | ¿Uso comercial? |
|---|---|---|---|---|
| **AI Motion Tracker** (este add-on) | Integración en Blender | GPL-3.0-or-later | — | ✅ Sí |
| **BootsTAPIR** (Google DeepMind, [tapnet](https://github.com/google-deepmind/tapnet)) | Seguimiento de puntos (cámara y marcadores) | Apache-2.0 (vendorizado en `third_party/tapir`) | Apache-2.0 | ✅ Sí |
| **MediaPipe** Pose / Face Landmarker (Google) | Cuerpo, cabeza, expresiones | Apache-2.0 | Apache-2.0 | ✅ Sí |
| **MediaPipe** Image Segmenter, DeepLab v3 (Google) | Ignorar objetos móviles | Apache-2.0 | Apache-2.0 | ✅ Sí |
| PyTorch | Ejecutar BootsTAPIR | BSD-3-Clause | — | ✅ Sí |
| OpenCV (≥ 4.5) | Video, esquinas, RANSAC | Apache-2.0 | — | ✅ Sí |
| dm-tree, einshape (Google) | Dependencias de BootsTAPIR | Apache-2.0 | — | ✅ Sí |
| NumPy | Cálculo | BSD-3-Clause | — | ✅ Sí |
| libmv (solver de cámara de Blender) | Solve de cámara | GPL (parte de Blender) | — | ✅ Sí |

## AI 3D Generator

| Componente | Para qué | Código | Pesos / modelos | ¿Uso comercial? |
|---|---|---|---|---|
| **AI 3D Generator** (este add-on) | Integración en Blender | GPL-3.0-or-later | — | ✅ Sí |
| **TripoSR** (Tripo AI + Stability AI, [repo](https://github.com/VAST-AI-Research/TripoSR)) | Imagen → 3D | MIT (vendorizado y adaptado en `ai_3d_generator/third_party/triposr`) | MIT ([stabilityai/TripoSR](https://huggingface.co/stabilityai/TripoSR)) | ✅ Sí |
| DINO ViT-B/16 (Meta), dentro de TripoSR | Codificador de imagen | Reimplementado (`vit.py`) | Apache-2.0, incluido en el checkpoint MIT de TripoSR | ✅ Sí |
| **IS-Net / DIS** (Xuebin Qin), ONNX de [rembg](https://github.com/danielgatis/rembg) | Quitar el fondo | Preprocesado propio | Apache-2.0 | ✅ Sí |
| **Stable Diffusion XL 1.0** (Stability AI) | Texto → imagen (opcional) | diffusers, Apache-2.0 | CreativeML Open RAIL++-M | ✅ Sí, con restricciones de uso (ver abajo) |
| **FLUX.1 [schnell]** (Black Forest Labs) | Texto → imagen (opcional) | diffusers, Apache-2.0 | Apache-2.0 | ✅ Sí |
| diffusers, transformers, accelerate, safetensors (Hugging Face) | Ejecutar SDXL / FLUX | Apache-2.0 | — | ✅ Sí |
| sentencepiece (Google), protobuf (Google) | Tokenizador de FLUX | Apache-2.0 / BSD-3-Clause | — | ✅ Sí |
| PyTorch, scikit-image, NumPy, SciPy | Cálculo, marching cubes | BSD-3-Clause | — | ✅ Sí |
| einops | Tensores | MIT | — | ✅ Sí |
| ONNX Runtime (Microsoft) | Ejecutar IS-Net | MIT | — | ✅ Sí |
| Pillow | Imágenes | MIT-CMU (HPND) | — | ✅ Sí |
| Blender (bmesh, Decimate, Remesh, QuadriFlow, Cycles, exportadores) | Optimización y exportación | GPL (parte de Blender) | — | ✅ Sí |

### Modelos 3D descartados a propósito

| Modelo | Motivo |
|---|---|
| Hunyuan3D 2.0 / 2.1 (Tencent) | Licencia comunitaria que **excluye la UE, Reino Unido y Corea del Sur** y limita usuarios |
| Stable Fast 3D, SPAR3D (Stability AI) | Stability Community License: gratis solo por debajo de 1 M$ de ingresos anuales |
| Zero123++ / InstantMesh | Pesos de Zero123++ **CC-BY-NC** (InstantMesh los hereda) |
| TRELLIS (Microsoft) | El modelo es MIT, pero su exportación a GLB depende de **nvdiffrast** (licencia NVIDIA no comercial) y de extensiones CUDA difíciles de instalar en Windows/Mac |
| TripoSG (VAST) | MIT, pero necesita GPU CUDA y su script usa **RMBG-1.4** (no comercial); candidato a un motor futuro con IS-Net |
| RMBG-1.4 / 2.0 (BRIA) | Pesos **no comerciales** (se usa IS-Net) |
| SDXL Turbo, SD3 / SD3.5 | Licencias no comerciales o comunitarias con límite de ingresos |

### Restricciones de SDXL (Open RAIL++-M)

Puedes vender y usar comercialmente lo que generes. La licencia prohíbe ciertos **usos**
(Anexo A: incumplir la ley, dañar a menores, desinformar, discriminar, suplantar a
personas…). Si redistribuyes el modelo SDXL en sí, esas restricciones deben acompañarlo.
Si prefieres una licencia sin restricciones de uso, elige FLUX.1 [schnell] (Apache-2.0).

### Tus propias imágenes

Si reconstruyes un objeto a partir de una foto o ilustración, el resultado deriva de esa
imagen: necesitas derechos sobre ella, y el objeto no debe reproducir marcas, logotipos,
personajes o diseños protegidos. El add-on guarda en cada exportación un
`<nombre>_licencias.json` con los modelos usados, sus licencias, el prompt y la semilla.

## Qué implica cada licencia

- **GPL-3.0 de los add-ons**: es obligatoria en la práctica para cualquier add-on de Blender
  (usa la API `bpy`, que es GPL). **No limita el uso comercial**: puedes usarlo en
  proyectos pagados, en un estudio, e incluso venderlo. Lo que produces con él (tracks,
  cámaras, animaciones, renders) es tuyo y no queda afectado por la GPL. Si
  *redistribuyes el add-on* (modificado o no), debes hacerlo con su código fuente bajo GPL.
- **MIT (TripoSR, ONNX Runtime, einops…)**: permite uso comercial, modificación y
  redistribución conservando el aviso de copyright (ver
  `ai_3d_generator/third_party/triposr/LICENSE`; los cambios están descritos en su
  `__init__.py`).
- **Apache-2.0 (BootsTAPIR, MediaPipe, OpenCV, IS-Net, FLUX.1 schnell…)**: permite uso comercial, modificación y
  redistribución. Al redistribuir hay que conservar el aviso de licencia y marcar los
  cambios (ver `ai_motion_tracker/third_party/tapir/LICENSE` y la nota en
  `tapir_model.py`). Apache-2.0 es compatible con GPL-3.0; por eso el add-on es
  GPL-3.0-*or-later* y no GPL-2.0-only.

## Modelos de tracking descartados a propósito

Son muy populares, pero **sus licencias prohíben o complican el uso comercial**:

| Modelo | Motivo |
|---|---|
| CoTracker / CoTracker3 (Meta) | Pesos y código **CC-BY-NC 4.0** (no comercial) |
| SuperPoint / SuperGlue (Magic Leap) | Licencia **no comercial** |
| Ultralytics YOLO | **AGPL-3.0** (obliga a publicar el código de servicios que lo usen) o licencia de pago |
| SMPL / SMPL-X y derivados (VIBE, 4DHumans…) | Modelos corporales con licencia **no comercial** |
| Segment Anything 2 (Meta) | Apache-2.0, pero el add-on no lo necesita; se prefirió MediaPipe para mantener las dependencias pequeñas |

## Descargas en tiempo de ejecución

Los modelos no se incluyen en los zips: se descargan la primera vez desde sus fuentes
oficiales. AI Motion Tracker: servidores de Google (`storage.googleapis.com/dm-tapnet` y
`…/mediapipe-models`), lista en `ai_motion_tracker/core/downloads.py`. AI 3D Generator:
Hugging Face (`stabilityai/TripoSR`, `stabilityai/stable-diffusion-xl-base-1.0`,
`black-forest-labs/FLUX.1-schnell`) y las releases de rembg en GitHub (IS-Net), lista en
`ai_3d_generator/core/downloads.py`.

> Esto es un resumen técnico, no asesoramiento legal. Si tu caso es delicado (por
> ejemplo, redistribuir el add-on dentro de un producto de pago), revisa los textos
> completos de las licencias.
