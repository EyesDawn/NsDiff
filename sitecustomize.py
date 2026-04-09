"""Runtime compatibility shims for the local research environment.

This repo depends on `pts`, which still imports
`gluonts.torch.modules.distribution_output`. In the installed
`gluonts==0.16.2`, that module was split across:

- gluonts.torch.distributions.distribution_output
- gluonts.torch.distributions.output
- gluonts.torch.modules.lambda_layer

Expose the old import path so existing experiment code can run without
patching the environment package in-place.
"""

from __future__ import annotations

import importlib
import sys
import types


def _install_gluonts_distribution_output_shim() -> None:
    module_name = "gluonts.torch.modules.distribution_output"

    try:
        importlib.import_module(module_name)
        return
    except ModuleNotFoundError:
        pass

    try:
        distribution_output = importlib.import_module(
            "gluonts.torch.distributions.distribution_output"
        )
        output = importlib.import_module("gluonts.torch.distributions.output")
        lambda_layer = importlib.import_module("gluonts.torch.modules.lambda_layer")
    except ModuleNotFoundError:
        return

    shim = types.ModuleType(module_name)
    shim.DistributionOutput = distribution_output.DistributionOutput
    shim.LambdaLayer = lambda_layer.LambdaLayer
    shim.PtArgProj = output.PtArgProj
    shim.__all__ = ["DistributionOutput", "LambdaLayer", "PtArgProj"]

    sys.modules[module_name] = shim


_install_gluonts_distribution_output_shim()
