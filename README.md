# AI Motion Tracker — add-on de Blender

Motion tracking hecho por IA dentro de Blender, usando solo modelos **open source con
licencia apta para uso comercial** (Apache-2.0 / BSD; detalle en [LICENSES.md](LICENSES.md)).

| Herramienta | IA | Resultado en Blender |
|---|---|---|
| **Tracking de cámara** | BootsTAPIR (Google DeepMind) siembra y sigue cientos de puntos; MediaPipe DeepLab ignora personas, coches y animales | Tracks en el Clip Editor → solve de cámara (libmv), con búsqueda automática de focal y limpieza de tracks malos |
| **Seguir marcadores seleccionados** | BootsTAPIR hacia delante y hacia atrás desde el frame actual | Tus marcadores, trackeados a través de oclusiones, desenfoque y cambios de luz |
| **Captura de cuerpo** | MediaPipe Pose (33 puntos, 2D + 3D) | Tracks 2D, empties 3D animados y un esqueleto (armature) listo para bakear/retargetear |
| **Captura facial** | MediaPipe Face (478 puntos) | Movimiento de cabeza 3D, 52 expresiones (blendshapes estilo ARKit) y tracks faciales 2D |

Compatible con **Blender 4.2 LTS → 5.x** (probado en 4.2, 4.5 y 5.0; pensado para 5.2 LTS).

## Instalación

1. Genera el zip (o descárgalo de las releases):
   ```bash
   python scripts/build_extension.py      # crea dist/ai_motion_tracker-1.0.0.zip
   ```
2. En Blender: **Edit › Preferences › Get Extensions › ▾ (arriba a la derecha) › Install from Disk…**
   y elige el zip.
3. Blender pedirá permiso de **red** (para descargar librerías y modelos): acéptalo.
4. En las preferencias del add-on pulsa **Instalar** en cada grupo:
   - **OpenCV** — siempre necesario.
   - **PyTorch** — para tracking de cámara y de marcadores.
     Antes de instalar elige *Variante de PyTorch*:
     - Windows + NVIDIA → **CUDA 12.8** (RTX 20xx en adelante, incluidas RTX 50xx).
     - macOS (Apple Silicon) o Linux → **Por defecto**.
     - Sin GPU → **Solo CPU** (funciona, pero es lento).
   - **MediaPipe** — cuerpo, cara y la máscara de objetos móviles.
5. Reinicia Blender si te lo pide.

Las librerías se instalan en una carpeta privada del add-on (no tocan el Python de
Blender) y los modelos se descargan la primera vez que se usan (~210 MB BootsTAPIR,
3–30 MB los de MediaPipe).

## Uso

Abre tu video en el **Movie Clip Editor** y pulsa **N** → pestaña **IA Tracking**.

### Tracking de cámara
1. En *Cámara › Solve* pon el **ancho de sensor** real (p. ej. 36 mm full-frame, 23.5 mm APS-C)
   y la **focal** si la sabes; si no, activa **Buscar focal automáticamente**.
2. Pulsa **Tracking de cámara con IA**. Puedes seguir trabajando; **ESC** cancela.
3. Al terminar verás los tracks (color verde) y el error del solve. Menos de ~0.5 px es
   un buen solve. Activa *Configurar escena de tracking* para crear cámara, fondo y suelo.

Cómo funciona: cada *N* frames se siembran puntos en esquinas (evitando objetos móviles),
BootsTAPIR los sigue hacia delante y hacia atrás dentro de una ventana, se descartan los
que no cumplen la geometría de la cámara (RANSAC epipolar) y se eligen los mejores para
tener *X* tracks en cada frame.

| Ajuste | Cuándo tocarlo |
|---|---|
| Calidad IA 512 / 256 | 512: sub-píxel (recomendado con GPU). 256: ~4× más rápido |
| Tracks por frame | Súbelo (80–150) si el solve falla o hay pocos puntos en algunos frames |
| Ventana / Resembrar cada | Ventanas más largas dan tracks más largos pero usan más memoria |
| Ignorar objetos móviles | Desactívalo si trackeas un plano de coches/personas *como objeto* |
| Consistencia mínima | Súbela (0.8–0.9) en planos con mucho movimiento en la escena |

### Seguir marcadores seleccionados
Coloca marcadores normales (Ctrl+clic), selecciónalos, ve a un frame donde estén bien
posicionados y pulsa **Seguir marcadores seleccionados con IA**. Se trackean en ambas
direcciones y los frames ocluidos quedan desactivados.

### Cuerpo y cara
En *Cuerpo y cara*:
- **Captura de cuerpo con IA** → colección *AI Pose*: empties por articulación,
  `AI_Pose_Rig` (huesos con *Copy Location* + *Stretch To*) y tracks `AIPose_*`.
  Para usarlo en otro rig: selecciona el armature › *Pose › Animation › Bake Action*
  (Visual Keying) y retargetea. Las posiciones 3D son relativas a la cadera (sin
  desplazamiento por la escena).
- **Captura facial con IA** → colección *AI Face*: `AI_Face_Cam` (con el clip de fondo) y
  `AI_Head`, hijo de esa cámara, con la rotación/posición de la cabeza y 52 propiedades
  animadas (`jawOpen`, `eyeBlinkLeft`…). Úsalas como drivers de shape keys.
  Para ver la cabeza alineada con el video, pon la resolución de la escena igual a la del clip.

## Rendimiento orientativo

Medido en CPU (sin GPU) sobre 40 frames 640×360: ~40 s con calidad 256 y ~140 s con 512.
Con una GPU NVIDIA reciente es del orden de 10–30× más rápido. Cuerpo y cara con
MediaPipe van a varios frames por segundo incluso en CPU.

## Precisión verificada

Sobre un clip sintético con movimiento de cámara conocido (ground truth):

| | Error mediano | p95 |
|---|---|---|
| Tracks de cámara (512) | 0.35 px | 0.80 px |
| Tracks de cámara (256) | 0.46 px | 1.10 px |
| Marcadores seleccionados, bidireccional (512) | 0.34 px | — |
| Solve de cámara con focal automática | 0.27 px de reproyección, 40/40 frames resueltos | |

## Problemas frecuentes

- **"Faltan dependencias"** → Preferencias del add-on › Instalar. Si pip falla, el log
  aparece ahí mismo (suele ser falta de conexión o un proxy).
- **El solve falla** → más *Tracks por frame*, activa *Buscar focal*, revisa el ancho de
  sensor. Planos con solo rotación (trípode) no tienen paralaje: usa *Tripod* en el
  solver de Blender.
- **Muy lento** → Calidad IA 256, *Solo rango de la escena*, o instala PyTorch con CUDA.
- **"MediaPipe no disponible"** en Linux sin escritorio → instala `libegl1` y `libgles2`.

## Desarrollo

```
ai_motion_tracker/
├── __init__.py, blender_manifest.toml
├── operators.py, ui.py, props.py, prefs.py   # Blender (hilo principal)
├── jobs.py          # hilo de trabajo + operador modal (progreso, ESC)
├── builders.py      # escribe tracks, keyframes, rigs
├── solve.py         # solve de cámara + búsqueda de focal (sección áurea)
├── deps.py          # pip a carpeta privada, numpy fijado al de Blender
├── core/            # sin bpy: IA, filtros, video (testeable fuera de Blender)
└── third_party/tapir/   # BootsTAPIR PyTorch (Apache-2.0, Google DeepMind)
```

Tests:
```bash
pip install numpy opencv-python pytest torch dm-tree einshape
python -m pytest tests -q
# precisión con el modelo real:
AIMT_BOOTSTAPIR=/ruta/bootstapir_checkpoint_v2.pt python -m pytest tests -q
```

Licencia: GPL-3.0-or-later. Componentes de terceros: ver [LICENSES.md](LICENSES.md).
