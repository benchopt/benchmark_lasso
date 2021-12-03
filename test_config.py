import sys

import pytest


def check_test_solver_install(solver_class):

    if solver_class.name.lower() == 'cyanure' and sys.platform == 'darwin':
        pytest.xfail('Cyanure is not easy to install on macos.')

    # Skip test_solver_install for julia in OSX as there is a version
    # conflict with conda packages for R
    # See issue benchopt/benchopt#64, PR benchopt/benchopt#252
    if 'julia' in solver_class.name.lower():
        pytest.xfail('Julia install from conda fails now.')

    # Lightning install is broken on python3.9+.
    # See issue scikit-learn-contrib/lightning#153.
    if (solver_class.name.lower() == 'lightning'
            and sys.version_info >= (3, 9)):
        pytest.xfail('Lightning install is broken on python3.9+.')

    # ModOpt install change numpy version, breaking celer install.
    # See CEA-COSMIC/ModOpt#144. Skipping for now
    if ('modopt' in solver_class.name.lower()):
        pytest.skip(
            'Modopt breaks other package installation by changing '
            'numpy version. Skipping for now.'
        )
