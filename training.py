import argparse
import yaml
from helper import Helper
from datetime import datetime
from tqdm import tqdm

from utils.utils import *
logger = logging.getLogger('logger')

def test(hlpr: Helper, epoch, backdoor=False):
    pass

def run_fl_round(hlpr: Helper, epoch):
    global_model = hlpr.task.model
    local_model = hlpr.task.local_model
    round_participants = hlpr.task.sample_users_for_round(epoch)
    weight_accumulator = hlpr.task.get_empty_accumulator()
    
    logger.info(f"Round {epoch} participants: {[user.user_id for user in round_participants]}")

    for user in tqdm(round_participants):
        hlpr.task.copy_params(global_model, local_model)
        optimizer = hlpr.task.make_optimizer(local_model)
        if user.compromised:
            
            if not user.user_id == 0:
                continue
            print(f"Compromised user: {user.user_id} in run_fl_round {epoch}")
            # for local_epoch in tqdm(range(hlpr.params.fl_poison_epochs)):
            #     train(hlpr, local_epoch, local_model, optimizer,
            #             user.train_loader, attack=True, global_model=global_model)
        else:
            print(f"Non-compromised user: {user.user_id} in run_fl_round {epoch}")
            # for local_epoch in range(hlpr.params.fl_local_epochs):
            #     train(hlpr, local_epoch, local_model, optimizer,
            #             user.train_loader, attack=False)
        
        # local_update = hlpr.attack.get_fl_update(local_model, global_model)
        # hlpr.save_update(model=local_update, userID=user.user_id)
        # if user.compromised:
        #     hlpr.attack.local_dataset = deepcopy(user.train_loader)
    

def run(hlpr: Helper):
    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        logger.info(f"Communication round {epoch}")
        run_fl_round(hlpr, epoch)
        metric = test(hlpr, epoch, backdoor=False)
        exit(0)
        # hlpr.record_accuracy(metric, test(hlpr, epoch, backdoor=True), epoch)

        # hlpr.save_model(hlpr.task.model, epoch, metric)
        

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
    
    
    try:
        run(helper)
    except Exception as e:
        print(e)
    
    