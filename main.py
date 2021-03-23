import os
from parser import Parser
from datetime import datetime

from misc.utils import *
from config import *

def main(args):

    #######################
    args = set_config(args)
    #######################
    
    if args.work_type == 'gen_data':
        from data.generator import DataGenerator
        dgen = DataGenerator(args)
        dgen.generate_data()

    elif args.work_type == 'train':

        os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
        
        now = datetime.now().strftime("%Y%m%d-%H%M")
        args.log_dir = f'{args.output_path}/logs/{now}-{args.model}-{args.task}'
        args.check_pts = f'{args.output_path}/check_pts/{now}-{args.model}-{args.task}'

        if args.model == 'fedmatch': 
            from models.fedmatch.server import Server
        else:
            print('incorrect model was given: {}'.format(args.model))
            os._exit(0)

        server = Server(args)
        server.run()
    
if __name__ == '__main__':
    main(Parser().parse())
