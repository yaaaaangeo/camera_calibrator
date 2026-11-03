"""Windows owned-task lifetime. No handles are inherited by P1 grandchildren.

Reference: https://learn.microsoft.com/windows/win32/procthread/job-objects
"""
import ctypes
from ctypes import wintypes as W
import os
import uuid


def _api():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    signatures = {
        'CreateJobObjectW': ([ctypes.c_void_p, W.LPCWSTR], W.HANDLE),
        'OpenJobObjectW': ([W.DWORD, W.BOOL, W.LPCWSTR], W.HANDLE),
        'SetInformationJobObject': ([W.HANDLE, ctypes.c_int, ctypes.c_void_p, W.DWORD], W.BOOL),
        'AssignProcessToJobObject': ([W.HANDLE, W.HANDLE], W.BOOL),
        'IsProcessInJob': ([W.HANDLE, W.HANDLE, ctypes.POINTER(W.BOOL)], W.BOOL),
        'GetCurrentProcess': ([], W.HANDLE),
        'CloseHandle': ([W.HANDLE], W.BOOL),
        'TerminateJobObject': ([W.HANDLE, W.UINT], W.BOOL),
    }
    for name, (args, result) in signatures.items():
        fn = getattr(kernel, name)
        fn.argtypes, fn.restype = args, result
    return kernel


def _assign(kernel, job, process):
    inside = W.BOOL()
    if not kernel.IsProcessInJob(process, job, ctypes.byref(inside)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not inside.value and not kernel.AssignProcessToJobObject(job, process):
        raise ctypes.WinError(ctypes.get_last_error())


class WindowsJob:
    def __init__(self):
        self.kernel = _api()
        self.name = 'Local\\CameraCalibratorP2_' + uuid.uuid4().hex
        self.handle = self.kernel.CreateJobObjectW(None, self.name)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        class Basic(ctypes.Structure):
            _fields_ = [('process_time', ctypes.c_longlong), ('job_time', ctypes.c_longlong),
                        ('flags', W.DWORD), ('min_ws', ctypes.c_size_t), ('max_ws', ctypes.c_size_t),
                        ('active', W.DWORD), ('affinity', ctypes.c_size_t),
                        ('priority', W.DWORD), ('scheduling', W.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in ('read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes')]
        class Extended(ctypes.Structure):
            _fields_ = [('basic', Basic), ('io', IO), ('process_memory', ctypes.c_size_t),
                        ('job_memory', ctypes.c_size_t), ('peak_process', ctypes.c_size_t),
                        ('peak_job', ctypes.c_size_t)]
        info = Extended()
        info.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            error = ctypes.get_last_error()
            self.close()
            raise ctypes.WinError(error)

    def assign(self, process):
        _assign(self.kernel, self.handle, W.HANDLE(int(process._handle)))

    def terminate(self):
        if self.handle and not self.kernel.TerminateJobObject(self.handle, 2):
            raise ctypes.WinError(ctypes.get_last_error())
        self.close()

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def join_job_from_environment():
    """Child self-enrollment closes the venv-launcher spawn/assignment race."""
    name = os.environ.get('CAMERA_P2_JOB')
    if os.name != 'nt' or not name:
        return
    kernel = _api()
    handle = kernel.OpenJobObjectW(0x0001 | 0x0004, False, name)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        _assign(kernel, handle, kernel.GetCurrentProcess())
    finally:
        kernel.CloseHandle(handle)
