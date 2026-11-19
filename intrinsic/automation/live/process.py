"""Own only the P2 task process tree; cancellation never touches unrelated apps."""
from __future__ import annotations
import os
from pathlib import Path
import signal
import subprocess
import sys


def process_alive(pid):
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def stop_tree(pid):
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'], capture_output=True,
                       creationflags=subprocess.CREATE_NO_WINDOW, timeout=10)
    else:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


class TaskProcess:
    def __init__(self, task):
        self.job = None
        if os.name == 'nt':
            from automation.live.windows_job import WindowsJob
            self.job = WindowsJob()
        self.log = open(Path(task['request']).parent / 'task.log', 'w', encoding='utf-8')
        env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
        if self.job is not None:
            env['CAMERA_P2_JOB'] = self.job.name
        import automation
        root = str(Path(automation.__file__).resolve().parent.parent)
        env['PYTHONPATH'] = os.pathsep.join(filter(None, [root, env.get('PYTHONPATH')]))
        options = ({'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
                   if os.name == 'nt' else {'start_new_session': True})
        try:
            self.process = subprocess.Popen([sys.executable, '-m', 'automation.live.task', task['request'],
                                             task['response'], str(os.getpid())], env=env,
                                            stdout=self.log, stderr=subprocess.STDOUT, **options)
            if self.job is not None:
                self.job.assign(self.process)
        except BaseException:
            if self.job is not None:
                self.job.close()
            self.log.close()
            raise

    def poll(self):
        return self.process.poll()

    def stop(self):
        if self.poll() is None:
            if self.job is not None:
                self.job.terminate()
            else:
                stop_tree(self.process.pid)
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        self.close()

    def close(self):
        if self.job is not None:
            self.job.close()
        self.log.close()
