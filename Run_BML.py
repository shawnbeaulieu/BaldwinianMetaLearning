#!/usr/bin/env python3.6
# Run_BML.py
# Author: Shawn Beaulieu
# June 3rd, 2017

import sys
import glob
import numpy as np
import pandas as pd
from functools import partial
from BaldwinianMetaLearning import NES

if __name__ == '__main__':

    Parameters = {

        'popsize': 100,
        'generations': 1000,
        'ranked_fitness': True,
        'dataset': 'omniglot',
        'num_classes': 5,
        'num_inst': 5,
        'learning_rate': 0.1,
        'inner_step_size': 0.1,
        'num_input_channels': 3,
        'inner_batch_size': 200,
        'meta_batch_size': 2,
        'num_updates': 2,
        'folder': 'data',
        'seed': int(sys.argv[1]),
        'folder': 'BML_Repo_Ranked_ShortQ'
    }


NES(Parameters)
