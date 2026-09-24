# Add-ons de IA para Blender

Add-ons que llevan IA a Blender usando solo modelos **open source con licencia apta para
uso comercial** (código **y** pesos). Detalle de licencias en [LICENSES.md](LICENSES.md).

| Add-on | Qué hace | Documentación |
|---|---|---|
| **AI 3D Generator** | Texto o imagen → modelo 3D (SDXL/FLUX + TripoSR) y Blender lo deja listo para producción: malla limpia y cerrada, tamaño real, decimado o quads, UVs, texturas PBR horneadas, LODs, exportación GLB/FBX/OBJ/USDZ/STL con ficha de licencias | [ai_3d_generator/README.md](ai_3d_generator/README.md) |
| **AI Motion Tracker** | Tracking de cámara (BootsTAPIR), seguimiento de marcadores, captura de cuerpo y cara (MediaPipe) | [ai_motion_tracker/README.md](ai_motion_tracker/README.md) |

Ambos son compatibles con **Blender 4.2 LTS → 5.x** y se pueden manejar desde Claude a
través de un servidor MCP para Blender (`import ai_3d_generator_api` /
`import ai_motion_tracker_api`).

## Instalación rápida

```bash
python scripts/build_extension.py                   # crea dist/<add-on>-<versión>.zip de ambos
python scripts/build_extension.py ai_3d_generator   # o solo uno
```

En Blender: **Edit › Preferences › Get Extensions › ▾ › Install from Disk…**, elige el zip,
acepta el permiso de red y pulsa **Instalar** en las dependencias desde las preferencias
del add-on. Las librerías de Python y los modelos se descargan a una carpeta privada del
add-on; no tocan el Python de Blender.

## Tests

```bash
pip install numpy opencv-python pytest torch dm-tree einshape          # AI Motion Tracker
pip install einops scikit-image onnxruntime pillow "bpy==5.2.0"        # AI 3D Generator (Python 3.13)
python -m pytest tests -q
```
Detalles de cada suite en la documentación de cada add-on.

Licencia: GPL-3.0-or-later. Componentes de terceros: ver [LICENSES.md](LICENSES.md).
