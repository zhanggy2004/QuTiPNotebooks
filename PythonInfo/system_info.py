from __future__ import print_function
import sys
import os
from importlib import import_module
import inspect
import platform
import time
from collections import OrderedDict

packages = ['IPython', 'Cython', 'numpy', 'scipy', 'matplotlib', 'qutip']


def info():
    entries = OrderedDict()
    entries['Python'] = sys.version.split('(')[0]
    entries['Build'] = ', '.join(platform.python_build())
    entries['Compiler'] = platform.python_compiler()

    for package in packages:
        try:
            mod = import_module(package)
            entries[package] = mod.__version__
        except:
            entries[package] = 'None'

    entries['User'] = os.getenv('USER')
    entries['OS'] = '{0} ({1})'.format(platform.system(), platform.machine())
    entries['Computer'] = platform.node()

    try:
        import qutip
        entries['QuTiP location'] = os.path.dirname(os.path.dirname(inspect.getsourcefile(qutip)))
    except:
        entries['QuTiP location'] = 'None'

    entries['Time'] = time.strftime('%a %b %d %H:%M:%S %Y %Z')

    return entries


if __name__ == '__main__':
    entries = info()
    maxlen = max(len(key) for key in entries)
    for (key, content) in entries.items():
        print('{0:{width}}{1}'.format(key+':', content, width=maxlen+2))
