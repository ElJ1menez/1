# SPDX-License-Identifier: GPL-3.0-or-later
"""One background job at a time: heavy AI work runs on a thread, results are
applied to Blender from a modal operator on the main thread."""

import inspect
import threading
import traceback

import bpy


class JobCancelled(Exception):
    pass


class Job:
    def __init__(self, name):
        self.name = name
        self.progress = 0.0
        self.message = "Iniciando…"
        self.error = None
        self.result = None
        self.cancelled = False
        self.thread_done = False
        self.finished = False
        self.lines = []

    def set(self, progress=None, message=None):
        if progress is not None:
            self.progress = float(progress)
        if message:
            self.message = message

    def check(self):
        if self.cancelled:
            raise JobCancelled()

    def log(self, line):
        self.lines.append(line)
        del self.lines[:-300]


_current = None
last_report = {"text": "", "level": "INFO"}
last_log = []


def current():
    return _current


def busy():
    return _current is not None and not _current.finished


def _run(job, worker, params):
    try:
        job.result = worker(job, params)
    except JobCancelled:
        job.cancelled = True
    except Exception as exc:  # reported to the user on the main thread
        job.error = "%s: %s" % (type(exc).__name__, exc)
        job.log(traceback.format_exc())
    finally:
        job.thread_done = True


def redraw_all():
    wm = bpy.context.window_manager
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type in {"CLIP_EDITOR", "VIEW_3D", "PREFERENCES"}:
                area.tag_redraw()


class JobOperator:
    """Mixin for operators that run a worker thread.

    Subclasses call self.launch(context, name, worker, params) from execute()
    and implement finish(context, result) which may return a generator to
    spread main-thread work (e.g. several camera solves) over timer ticks.
    """

    def launch(self, context, name, worker, params):
        global _current
        if busy():
            self.report({"ERROR"}, "Ya hay un proceso de IA en marcha")
            return {"CANCELLED"}
        self._job = _current = Job(name)
        self._finisher = None
        threading.Thread(target=_run, args=(self._job, worker, params), daemon=True).start()
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.15, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        job = self._job
        if job.cancelled and not job.thread_done:
            job.set(message="Cancelando…")
        if event.type == "ESC" and event.value == "PRESS":
            job.cancelled = True
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        redraw_all()
        try:
            if self._finisher is not None:
                if job.cancelled:
                    return self._end(context, "WARNING", "Cancelado")
                next(self._finisher)
                return {"RUNNING_MODAL"}
            if not job.thread_done:
                return {"RUNNING_MODAL"}
            if job.cancelled:
                return self._end(context, "WARNING", "Cancelado")
            if job.error:
                return self._end(context, "ERROR", job.error)
            job.set(progress=1.0, message="Aplicando resultados en Blender")
            res = self.finish(context, job.result)
            if inspect.isgenerator(res):
                self._finisher = res
                return {"RUNNING_MODAL"}
            return self._end(context, "INFO", res or "Listo")
        except StopIteration as stop:
            return self._end(context, "INFO", stop.value or "Listo")
        except Exception as exc:
            job.log(traceback.format_exc())
            return self._end(context, "ERROR", "%s: %s" % (type(exc).__name__, exc))

    def _end(self, context, level, text):
        job = self._job
        job.finished = True
        context.window_manager.event_timer_remove(self._timer)
        last_report["text"], last_report["level"] = text, level
        last_log[:] = job.lines
        if level == "ERROR" and job.lines:
            print("[AI Motion Tracker]\n" + "\n".join(job.lines[-40:]))
        self.report({level}, text)
        redraw_all()
        return {"FINISHED"} if level != "ERROR" else {"CANCELLED"}

    def finish(self, context, result):
        return "Listo"


class AIMT_OT_cancel(bpy.types.Operator):
    bl_idname = "aimt.cancel"
    bl_label = "Cancelar"
    bl_description = "Cancela el proceso de IA en marcha"

    def execute(self, context):
        if _current is not None:
            _current.cancelled = True
        return {"FINISHED"}
