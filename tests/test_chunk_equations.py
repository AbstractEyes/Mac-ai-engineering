"""The chunked forward and backward equations of docs/prior_art_gated_delta.md (sections 1.4-1.5), checked in float64 on the CPU
against autograd through the token recurrence (tools/check_chunk_equations.py): the derivation the chunked kernels implement."""
import os
import re
import subprocess
import sys


def test_chunked_equations_match_the_recurrence():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "check_chunk_equations.py")
    out = subprocess.run([sys.executable, "-I", path], capture_output=True, text=True, check=True).stdout
    errs = [float(x) for x in re.findall(r"max err\s+([0-9.eE+-]+)", out)]
    assert len(errs) >= 8, out
    assert max(errs) < 1e-12, out
