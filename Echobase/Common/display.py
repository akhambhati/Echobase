'''
Display messages to the console or screen

Created by: Ankit Khambhati

Change Log
----------
2016/02/22 - Consolidate display code from other local modules
'''

import sys
import time
from IPython.display import clear_output


def my_display(txt, verbose, str_path=None):
    '''
    Print string to the screen

    Parameters
    ----------
        txt: str
            Text to be printed to the screen

        verbose: bool
            Option to robustly automate whether
            text gets printed to the screen

        str_path: str
            Path to store stdout
    '''

    if str_path is not None:
        old_stdout = sys.stdout
        sys.stdout = open(str_path, 'a+')

    if verbose:
        sys.stdout.write(txt)
        sys.stdout.flush()

    if str_path is not None:
        sys.stdout = old_stdout


def par_watch_stdout(ar, dt=1.0, truncate=1000):
    '''
    Print console status from multiple console outputs
    '''

    line_break = '-'*30

    while not ar.ready():
        clear_output()
        # Clear_output does not work in plain console
        print(line_break)
        print('%.3fs elapsed' % ar.elapsed)
        print(line_break)
        print('')
        for eid, stdout in zip(range(len(ar)), ar.stdout):
            if stdout:
                print('\n\n%s\n[ ipengine %2i  ]\n%s' % (
                    line_break, eid, stdout[-truncate:]))
        sys.stdout.flush()
        time.sleep(dt)
