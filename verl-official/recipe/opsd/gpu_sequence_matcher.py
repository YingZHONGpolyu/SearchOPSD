"""Launch the exact difflib-compatible kernel through the CUDA driver API."""

import ctypes
import difflib
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch


def build_kernel():
    """Compile a device-only cubin; it does not depend on PyTorch's C++ ABI."""
    source = Path(__file__).with_name('sequence_matcher.cu')
    binary = source.with_suffix('.cubin')
    if not binary.exists() or binary.stat().st_mtime_ns < source.stat().st_mtime_ns:
        # Prefer the server's recent toolkit, while allowing the CUDA toolkit in a container.
        candidates = [os.environ.get('OPSD_NVCC'), '/usr/local/cuda-12.9/bin/nvcc',
                      '/usr/local/cuda/bin/nvcc', shutil.which('nvcc')]
        compiler = next((path for path in candidates if path and Path(path).is_file()), None)
        if compiler is None:
            raise RuntimeError('CUDA nvcc is required to build the exact GPU miner')
        subprocess.run([compiler, '-O3', '-std=c++17', '--cubin',
                        '-arch=sm_80', str(source), '-o', str(binary)], check=True)
    return binary


class GPUSequenceMatcher:
    """Return the same character distances as core.character_distance, in GPU batches."""

    def __init__(self, device, binary):
        torch.cuda.set_device(device)
        torch.cuda.init()
        self.device = device
        self.driver = ctypes.CDLL('libcuda.so.1')
        self.module, self.function = ctypes.c_void_p(), ctypes.c_void_p()
        self.check(self.driver.cuModuleLoad(ctypes.byref(self.module), str(binary).encode()))
        self.check(self.driver.cuModuleGetFunction(ctypes.byref(self.function), self.module, b'sequence_matches'))
        self.driver.cuLaunchKernel.argtypes = [ctypes.c_void_p, *([ctypes.c_uint] * 7),
                                              ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f'CUDA driver call failed with code {code}')

    def distances(self, pairs):
        """Encode Unicode code points, run matching blocks, and preserve float tie-breaking."""
        torch.cuda.set_device(self.device)
        # UTF-32 code points match Python string indexing, including non-ASCII text.
        arrays_a = [np.frombuffer(a.encode('utf-32-le', 'surrogatepass'), dtype='<i4') for a, _ in pairs]
        arrays_b = [np.frombuffer(b.encode('utf-32-le', 'surrogatepass'), dtype='<i4') for _, b in pairs]
        offsets_a = np.concatenate(([0], np.cumsum([len(a) for a in arrays_a]))).astype(np.int64)
        offsets_b = np.concatenate(([0], np.cumsum([len(b) for b in arrays_b]))).astype(np.int64)
        tensors = [torch.from_numpy(np.concatenate(arrays_a)).to('cuda'),
                   torch.from_numpy(np.concatenate(arrays_b)).to('cuda'),
                   torch.from_numpy(offsets_a).to('cuda'), torch.from_numpy(offsets_b).to('cuda'),
                   torch.empty(len(pairs), dtype=torch.int32, device='cuda'),
                   torch.empty(len(pairs), dtype=torch.int32, device='cuda')]
        pointers = [ctypes.c_uint64(tensor.data_ptr()) for tensor in tensors]
        count = ctypes.c_int(len(pairs))
        arguments = (ctypes.c_void_p * 7)(*[ctypes.addressof(value) for value in [*pointers, count]])
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        self.check(self.driver.cuLaunchKernel(self.function, len(pairs), 1, 1, 256, 1, 1, 0,
                                             stream, arguments, None))
        torch.cuda.synchronize()
        matched = tensors[4].cpu().tolist()
        overflow = tensors[5].cpu().tolist()
        scores = []
        for index, (a, b) in enumerate(pairs):
            if overflow[index]:
                # A bounded-stack overflow must preserve correctness rather than approximate.
                scores.append(1.0 - difflib.SequenceMatcher(None, a, b, autojunk=False).ratio())
            else:
                length = len(a) + len(b)
                scores.append(1.0 - (2.0 * matched[index] / length if length else 1.0))
        return scores
