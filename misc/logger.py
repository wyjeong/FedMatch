__author__ = "Wonyong Jeong"
__email__ = "wyjeong@kaist.ac.kr"

from datetime import datetime

from misc.utils import *

class Logger:

    def __init__(self, args, client_id=None):
        """ Logging Module

        Created by:
            Wonyong Jeong (wyjeong@kaist.ac.kr)
        """
        self.args = args
        self.options = vars(self.args)

    def print(self, client_id, message):
        name = f'client-{client_id}' if type(client_id) == int else 'server'
        print(f'[{datetime.now().strftime("%Y/%m/%d-%H:%M:%S")}]'+
                f'[{self.args.model}]'+
                f'[{self.args.task}]'+
                f'[{name}] '+
                f'{message}')

    def save_current_state(self, client_id, current_state):
        current_state['options'] = self.options
        name = f'client-{client_id}' if type(client_id) == int else 'server'
        write_file(self.args.log_dir, f'{name}.txt', current_state)
