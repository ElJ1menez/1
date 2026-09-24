# SPDX-License-Identifier: GPL-3.0-or-later
"""Build dist/<addon>-<version>.zip, installable in Blender 4.2+
(Edit > Preferences > Get Extensions > ▾ > Install from Disk).

    python scripts/build_extension.py                    # every add-on
    python scripts/build_extension.py ai_3d_generator    # just one

Equivalent to `blender --command extension build`, but needs only Python.
"""

import os
import re
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADDONS = ("ai_motion_tracker", "ai_3d_generator")


def build(addon):
    src = os.path.join(ROOT, addon)
    manifest = open(os.path.join(src, "blender_manifest.toml"), encoding="utf-8").read()
    version = re.search(r'^version\s*=\s*"([^"]+)"', manifest, re.M).group(1)
    os.makedirs(os.path.join(ROOT, "dist"), exist_ok=True)
    out = os.path.join(ROOT, "dist", "%s-%s.zip" % (addon, version))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for folder, dirs, files in os.walk(src):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in files:
                if f.endswith((".pyc", ".zip")):
                    continue
                path = os.path.join(folder, f)
                z.write(path, os.path.relpath(path, src))  # manifest at zip root
        z.write(os.path.join(ROOT, "LICENSES.md"), "LICENSES.md")
    return out


def main(argv):
    names = argv or ADDONS
    unknown = sorted(set(names) - set(ADDONS))
    if unknown:
        sys.exit("Add-on desconocido: %s. Opciones: %s" % (", ".join(unknown), ", ".join(ADDONS)))
    for name in names:
        print(build(name))


if __name__ == "__main__":
    main(sys.argv[1:])
