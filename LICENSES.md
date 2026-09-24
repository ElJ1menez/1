# Licencias y uso comercial

Criterio del proyecto: **solo se usan componentes open source cuya licencia permite el
uso comercial**, tanto del código como de los **pesos de los modelos** de IA (que suelen
tener una licencia distinta a la del código, y es donde está la trampa habitual).

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

## Qué implica cada licencia

- **GPL-3.0 del add-on**: es obligatoria en la práctica para cualquier add-on de Blender
  (usa la API `bpy`, que es GPL). **No limita el uso comercial**: puedes usarlo en
  proyectos pagados, en un estudio, e incluso venderlo. Lo que produces con él (tracks,
  cámaras, animaciones, renders) es tuyo y no queda afectado por la GPL. Si
  *redistribuyes el add-on* (modificado o no), debes hacerlo con su código fuente bajo GPL.
- **Apache-2.0 (BootsTAPIR, MediaPipe, OpenCV…)**: permite uso comercial, modificación y
  redistribución. Al redistribuir hay que conservar el aviso de licencia y marcar los
  cambios (ver `ai_motion_tracker/third_party/tapir/LICENSE` y la nota en
  `tapir_model.py`). Apache-2.0 es compatible con GPL-3.0; por eso el add-on es
  GPL-3.0-*or-later* y no GPL-2.0-only.

## Modelos descartados a propósito

Son muy populares, pero **sus licencias prohíben o complican el uso comercial**:

| Modelo | Motivo |
|---|---|
| CoTracker / CoTracker3 (Meta) | Pesos y código **CC-BY-NC 4.0** (no comercial) |
| SuperPoint / SuperGlue (Magic Leap) | Licencia **no comercial** |
| Ultralytics YOLO | **AGPL-3.0** (obliga a publicar el código de servicios que lo usen) o licencia de pago |
| SMPL / SMPL-X y derivados (VIBE, 4DHumans…) | Modelos corporales con licencia **no comercial** |
| Segment Anything 2 (Meta) | Apache-2.0, pero el add-on no lo necesita; se prefirió MediaPipe para mantener las dependencias pequeñas |

## Descargas en tiempo de ejecución

Los modelos no se incluyen en el zip: se descargan la primera vez desde los servidores
oficiales de Google (`storage.googleapis.com/dm-tapnet` y `…/mediapipe-models`). La
lista exacta, con URL y licencia, está en `ai_motion_tracker/core/downloads.py`.

> Esto es un resumen técnico, no asesoramiento legal. Si tu caso es delicado (por
> ejemplo, redistribuir el add-on dentro de un producto de pago), revisa los textos
> completos de las licencias.
