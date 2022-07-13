import sys
import pytest

from benchopt.utils.sys_info import get_cuda_version


def check_test_solver_install(solver_class):

    if solver_class.name.lower() == 'cyanure' and sys.platform == 'darwin':
        pytest.xfail('Cyanure is not easy to install on macos.')

    # Skip test_solver_install for julia in OSX as there is a version
    # conflict with conda packages for R
    # See issue benchopt/benchopt#64, PR benchopt/benchopt#252
    if 'julia' in solver_class.name.lower():
        pytest.xfail('Julia install from conda fails currently.')

    if "spams" in solver_class.name.lower():
        pytest.skip("python-spams is not released for python 3.9 yet")

    if "cuml" in solver_class.name.lower():
        if sys.platform == "darwin":
            pytest.xfail("Cuml is not supported on MacOS.")
        cuda_version = get_cuda_version()
        if cuda_version is None:
            pytest.xfail("Cuml needs a working GPU hardware.")
