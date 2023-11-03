import argparse
import yaml
from helper import Helper
from datetime import datetime
from utils.utils import *
logger = logging.getLogger('logger')

def run(params):
    print(params)
    for epoch in range(params['epochs']):
        print(epoch)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backdoors')
    parser.add_argument('--params', dest='params', required=True)
    parser.add_argument('--name', dest='name', required=True)
    # python training.py --name mnist --params exps/mnist_fed.yaml
    
    args = parser.parse_args()
    print(args)
    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    # print(params)
    params['current_time'] = datetime.now().strftime('%b.%d_%H.%M.%S')
    params['name'] = args.name
    
    helper = Helper(params)
    
    # logger = create_logger()
    
    logger.info(create_table(params))
    # logger.info(f"Creating folder ")
    exit(0)
    
    try:
        run(params)
    except Exception as e:
        print(e)
    