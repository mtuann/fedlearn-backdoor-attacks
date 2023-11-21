import argparse
import yaml
from helper import Helper
from datetime import datetime
from tqdm import tqdm
import wandb

from utils.utils import *
logger = logging.getLogger('logger')

def train(hlpr: Helper, epoch, model, optimizer, train_loader, attack=False, global_model=None):
    criterion = hlpr.task.criterion
    model.train()
    # for i, data in tqdm(enumerate(train_loader)):
    for i, data in enumerate(train_loader):
        batch = hlpr.task.get_batch(i, data)
        model.zero_grad()
        loss = hlpr.attack.compute_blind_loss(model, criterion, batch, attack, global_model)
        loss.backward()
        optimizer.step()
        # print(f"Epoch {epoch} batch {i} loss {loss.item()}")

        if i == hlpr.params.max_batch_id:
            break
    # metric = hlpr.task.report_metrics(epoch,
    #                          prefix=f'Backdoor {str(backdoor):5s}. Epoch: ')
    return

def test(hlpr: Helper, epoch, backdoor=False, model=None):
    if model is None:
        model = hlpr.task.model
    model.eval()
    hlpr.task.reset_metrics()
    with torch.no_grad():
        for i, data in tqdm(enumerate(hlpr.task.test_loader)):
            batch = hlpr.task.get_batch(i, data)
            if backdoor:
                batch = hlpr.attack.synthesizer.make_backdoor_batch(batch,
                                                                    test=True,
                                                                    attack=True)

            outputs = model(batch.inputs)
            hlpr.task.accumulate_metrics(outputs=outputs, labels=batch.labels)
    metric = hlpr.task.report_metrics(epoch,
                             prefix=f'Backdoor {str(backdoor):5s}. Epoch: ')
    return metric

def run_fl_round(hlpr: Helper, epoch):
    global_model = hlpr.task.model
    local_model = hlpr.task.local_model
    round_participants = hlpr.task.sample_users_for_round(epoch)
    # hlpr.params.fl_round_participants = [user.user_id for user in round_participants]
    
    weight_accumulator = hlpr.task.get_empty_accumulator()
    
    logger.info(f"Round epoch {epoch} with participants: {[user.user_id for user in round_participants]} and weight: {hlpr.params.fl_weight_contribution}")
    # log number of sample per user
    logger.info(f"Round epoch {epoch} with participants sample size: {[user.number_of_samples for user in round_participants]}")
    
    for user in tqdm(round_participants):
        hlpr.task.copy_params(global_model, local_model)
        optimizer = hlpr.task.make_optimizer(local_model)
        if user.compromised:
            # if not user.user_id == 0:
            #     continue
            
            logger.warning(f"Compromised user: {user.user_id} in run_fl_round {epoch}")
            for local_epoch in tqdm(range(hlpr.params.fl_poison_epochs)): # fl_poison_epochs)):
                train(hlpr, local_epoch, local_model, optimizer,
                        user.train_loader, attack=True, global_model=global_model)
                
        else:
            logger.warning(f"Non-compromised user: {user.user_id} in run_fl_round {epoch}")
            for local_epoch in range(hlpr.params.fl_local_epochs):
                train(hlpr, local_epoch, local_model, optimizer,
                        user.train_loader, attack=False)
        
        local_update = hlpr.attack.get_fl_update(local_model, global_model)
        
        hlpr.save_update(model=local_update, userID=user.user_id)
        
        if user.compromised:
            hlpr.attack.perform_attack(global_model, user, epoch)
            # hlpr.attack.local_dataset = deepcopy(user.train_loader)
            
    # logger.info(f"Round {epoch} attack")
    # hlpr.attack.perform_attack(global_model, epoch)
    # logger.info(f"Round {epoch} aggregation")
    hlpr.defense.aggr(weight_accumulator, global_model)
    # logger.info(f"Round {epoch} update global model")
    hlpr.task.update_global_model(weight_accumulator, global_model)

def run(hlpr: Helper):
    metric = test(hlpr, -1, backdoor=False)
    logger.info(f"Before training main metric: {metric}")
        
    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        logger.info(f"Communication round {epoch}")
        run_fl_round(hlpr, epoch)
        metric = test(hlpr, epoch, backdoor=False)
        main_metric = hlpr.task.get_metrics()
        
        # logger.info(f"Epoch {epoch} main metric: {metric}")
        metric_bd = test(hlpr, epoch, backdoor=True)
        backdoor_metric = hlpr.task.get_metrics()
        
        
        wandb.log({'main_acc': main_metric['accuracy'], 'main_loss': main_metric['loss'], 
                  'backdoor_acc': backdoor_metric['accuracy'], 'backdoor_loss': backdoor_metric['loss']}, 
                  step=epoch)
        logger.info(f"Epoch {epoch} backdoor metric: {metric}")
        
        # hlpr.record_accuracy(metric, test(hlpr, epoch, backdoor=True), epoch)

        hlpr.save_model(hlpr.task.model, epoch, metric)
        

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
    # - Exp 01: fl_total_participants/ fl_no_models/ fl_number_of_adversaries: 200/ 20/ 4; fl_dirichlet_alpha: 0.5; lr: 0.02
    # - Exp 02: fl_total_participants/ fl_no_models/ fl_number_of_adversaries: 200/ 10/ 4; fl_dirichlet_alpha: 0.5; lr: 0.02
    # - Exp 03: fl_total_participants/ fl_no_models/ fl_number_of_adversaries: 100/ 10/ 4; fl_dirichlet_alpha: 0.5; lr: 0.02
    # - Exp 04: fl_total_participants/ fl_no_models/ fl_number_of_adversaries: 100/ 10/ 4; fl_dirichlet_alpha: 0.5; lr: 0.01
    # - Exp 05: fl_total_participants/ fl_no_models/ fl_number_of_adversaries: 100/ 10/ 4; fl_dirichlet_alpha: 0.5; lr: 0.05

    # logger.info(create_table(params))
    wandb.init(project="simple-backdoor-fl", entity="mtuann", name=f"{params['name']}-{params['current_time']}-{params['fl_total_participants']}-{params['fl_no_models']}-{params['fl_number_of_adversaries']}-{params['fl_dirichlet_alpha']}-{params['lr']}")
    
    try:
        run(helper)
    except Exception as e:
        print(e)
    
    