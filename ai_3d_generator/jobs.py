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
        self.level = None
        self.report = None

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
            if area.type in {"VIEW_3D", "PROPERTIES", "PREFERENCES"}:
                area.tag_redraw()


def start(name, worker, params):
    """Start `worker(job, params)` on a thread. Raises RuntimeError if busy."""
    global _current
    if busy():
        raise RuntimeError("Ya hay un proceso de IA en marcha: %s" % _current.name)
    job = _current = Job(name)
    threading.Thread(target=_run, args=(job, worker, params), daemon=True).start()
    return job


class Driver:
    """Main-thread state machine: wait for the thread, then run finish().

    finish(context, result) returns a message, or a generator that is
    advanced one step per tick so long main-thread work (remeshing, texture
    baking) never freezes the UI. tick() returns None while running and
    (level, text) once done.
    """

    def __init__(self, job, finish):
        self.job = job
        self.finish = finish
        self.finisher = None

    def tick(self, context):
        job = self.job
        if job.cancelled and not job.thread_done:
            job.set(message="Cancelando…")
        try:
            if self.finisher is not None:
                if job.cancelled:
                    return "WARNING", "Cancelado"
                next(self.finisher)
                return None
            if not job.thread_done:
                return None
            if job.cancelled:
                return "WARNING", "Cancelado"
            if job.error:
                return "ERROR", job.error
            job.set(progress=1.0, message="Aplicando resultados en Blender")
            res = self.finish(context, job.result)
            if inspect.isgenerator(res):
                self.finisher = res
                return None
            return "INFO", res or "Listo"
        except StopIteration as stop:
            return "INFO", stop.value or "Listo"
        except Exception as exc:
            job.log(traceback.format_exc())
            return "ERROR", "%s: %s" % (type(exc).__name__, exc)


def complete(job, level, text):
    job.finished = True
    job.level, job.report = level, text
    last_report["text"], last_report["level"] = text, level
    last_log[:] = job.lines
    if level == "ERROR" and job.lines:
        print("[AI 3D Generator]\n" + "\n".join(job.lines[-40:]))
    redraw_all()


def run_in_background(name, worker, params, finish):
    """Like the modal operators, but driven by bpy.app.timers: needs no
    window or event loop context, so it works from scripts and MCP calls."""
    job = start(name, worker, params)
    driver = Driver(job, finish)

    def timer():
        redraw_all()
        done = driver.tick(bpy.context)
        if done is None:
            return 0.15
        complete(job, *done)
        return None

    bpy.app.timers.register(timer, first_interval=0.15)
    return job


class JobOperator:
    """Mixin for operators that run a worker thread.

    Subclasses call self.launch(context, name, worker, params, finish) from
    execute(); see Driver for what finish() may return.
    """

    def launch(self, context, name, worker, params, finish):
        try:
            self._job = start(name, worker, params)
        except RuntimeError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self._driver = Driver(self._job, finish)
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.15, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "ESC" and event.value == "PRESS":
            self._job.cancelled = True
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        redraw_all()
        done = self._driver.tick(context)
        if done is None:
            return {"RUNNING_MODAL"}
        level, text = done
        context.window_manager.event_timer_remove(self._timer)
        complete(self._job, level, text)
        self.report({level}, text)
        return {"FINISHED"} if level != "ERROR" else {"CANCELLED"}


class AI3D_OT_cancel(bpy.types.Operator):
    bl_idname = "ai3d.cancel"
    bl_label = "Cancelar"
    bl_description = "Cancela el proceso de IA en marcha"

    def execute(self, context):
        if _current is not None:
            _current.cancelled = True
        return {"FINISHED"}
