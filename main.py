# -*- coding: UTF-8 -*-

import sys
sys.path.insert(0, '.')
from semi_tracker.interface import run
from semi_tracker.utils import logger

if __name__ == '__main__':
    logger.info('Start... \n:')
    run()
