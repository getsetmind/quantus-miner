#!/usr/bin/env python3
"""Compile the embedded CUDA kernel with NVRTC, without a GPU or mining.

Requires an NVIDIA NVRTC library and its matching builtins library. This checks
compiler compatibility only: it does not execute PTX or prove hash correctness.
"""

import argparse
import ctypes
import hashlib
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument(
        "--kernel",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "crates/engine-cuda/src/kernels/mining.cu",
    )
    parser.add_argument("--arch", default="compute_86")
    args = parser.parse_args()
    library_path = args.library.resolve(strict=True)
    dll_directory = (
        os.add_dll_directory(str(library_path.parent)) if os.name == "nt" else None
    )
    try:
        lib = ctypes.CDLL(str(library_path))
        program_type = ctypes.c_void_p
        signatures = {
            "nvrtcVersion": [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)],
            "nvrtcCreateProgram": [
                ctypes.POINTER(program_type), ctypes.c_char_p, ctypes.c_char_p,
                ctypes.c_int, ctypes.POINTER(ctypes.c_char_p),
                ctypes.POINTER(ctypes.c_char_p),
            ],
            "nvrtcCompileProgram": [program_type, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
            "nvrtcGetProgramLogSize": [program_type, ctypes.POINTER(ctypes.c_size_t)],
            "nvrtcGetProgramLog": [program_type, ctypes.c_void_p],
            "nvrtcGetPTXSize": [program_type, ctypes.POINTER(ctypes.c_size_t)],
            "nvrtcGetPTX": [program_type, ctypes.c_void_p],
            "nvrtcDestroyProgram": [ctypes.POINTER(program_type)],
        }
        for name, argtypes in signatures.items():
            fn = getattr(lib, name)
            fn.argtypes = argtypes
            fn.restype = ctypes.c_int

        def checked(name, *values):
            result = getattr(lib, name)(*values)
            if result != 0:
                raise RuntimeError(f"{name} failed: NVRTC status {result}")

        major, minor = ctypes.c_int(), ctypes.c_int()
        checked("nvrtcVersion", ctypes.byref(major), ctypes.byref(minor))
        source = args.kernel.read_bytes()
        print(f"NVRTC {major.value}.{minor.value}; target {args.arch}", flush=True)
        print(f"Source SHA256: {hashlib.sha256(source).hexdigest()}", flush=True)
        program = program_type()
        checked("nvrtcCreateProgram", ctypes.byref(program), source, b"mining.cu", 0, None, None)
        try:
            options = (ctypes.c_char_p * 1)(f"--gpu-architecture={args.arch}".encode())
            status = lib.nvrtcCompileProgram(program, len(options), options)
            size = ctypes.c_size_t()
            checked("nvrtcGetProgramLogSize", program, ctypes.byref(size))
            log = ctypes.create_string_buffer(max(size.value, 1))
            checked("nvrtcGetProgramLog", program, log)
            if log.value:
                print(log.value.decode("utf-8", errors="replace"), flush=True)
            if status != 0:
                print(f"COMPILE FAILED: NVRTC status {status}")
                return 1
            checked("nvrtcGetPTXSize", program, ctypes.byref(size))
            ptx = ctypes.create_string_buffer(size.value)
            checked("nvrtcGetPTX", program, ptx)
            print(f"COMPILE PASSED: {size.value} PTX bytes; SHA256 {hashlib.sha256(ptx.raw).hexdigest()}")
            print("No GPU code executed; no pool connection or share submission.")
            return 0
        finally:
            checked("nvrtcDestroyProgram", ctypes.byref(program))
    finally:
        if dll_directory is not None:
            dll_directory.close()


if __name__ == "__main__":
    raise SystemExit(main())
