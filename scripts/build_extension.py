# SPDX-License-Identifier: GPL-3.0-or-later
"""Build dist/ai_motion_tracker-<version>.zip, installable in Blender 4.2+
(Edit > Preferences > Get Extensions > ▾ > Install from Disk).

Equivalent to `blender --command extension build`, but needs only Python.
"""

import os
import re
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "ai_motion_tracker")


def main():
    manifest = open(os.path.join(SRC, "blender_manifest.toml"), encoding="utf-8").read()
    version = re.search(r'^version\s*=\s*"([^"]+)"', manifest, re.M).group(1)
    os.makedirs(os.path.join(ROOT, "dist"), exist_ok=True)
    out = os.path.join(ROOT, "dist", "ai_motion_tracker-%s.zip" % version)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for folder, dirs, files in os.walk(SRC):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in files:
                if f.endswith((".pyc", ".zip")):
                    continue
                path = os.path.join(folder, f)
                z.write(path, os.path.relpath(path, SRC))  # manifest at zip root
        z.write(os.path.join(ROOT, "LICENSES.md"), "LICENSES.md")
    print(out)


if __name__ == "__main__":
    main()
