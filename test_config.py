import sys
import pytest

from benchopt.utils.sys_info import _get_cuda_version


def check_test_solver_install(solver_class):

    if solver_class.name.lower() == 'cyanure' and sys.platform == 'darwin':
        pytest.xfail('Cyanure is not easy to install on macos.')

    # Skip test_solver_install for julia in OSX as there is a version
    # conflict with conda packages for R
    # See issue benchopt/benchopt#64, PR benchopt/benchopt#252
    if 'julia' in solver_class.name.lower():
        pytest.xfail('Julia install from conda fails currently.')

    # ModOpt install change numpy version, breaking celer install.
    # See CEA-COSMIC/ModOpt#144. Skipping for now
    if ('modopt' in solver_class.name.lower()):
        pytest.skip(
            'Modopt breaks other package installation by changing '
            'numpy version. Skipping for now.'
        )

    if solver_class.name.lower() == "snapml[gpu=True]".lower():
        if _get_cuda_version() is None:
            pytest.skip("snapml[gpu=True] needs a GPU to run")
