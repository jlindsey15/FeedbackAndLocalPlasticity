import argparse
import copy
import logging

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch import optim
from torch.nn import functional as F

import datasets.task_sampler as ts
import model.modelfactory as mf
from experiment.experiment import experiment
from model.oja_meta_learner import MetaLearnerRegression

#
logger = logging.getLogger('experiment')


def construct_set(iterators, sampler, steps=2, iid=False):
    '''
    :param iterators: List of iterators to sample different tasks
    :param sampler: object that samples data from the iterator and appends task ids
    :param steps: no of batches per task
    :param iid:
    :return:
    '''
    x_spt = []
    y_spt = []
    for id, it1 in enumerate(iterators):
        for inner in range(steps):
            x, y = sampler.sample_batch(it1, id, args.minibatch_size)
            x_spt.append(x)
            y_spt.append(y)

    x_qry = []
    y_qry = []
    for id, it1 in enumerate(iterators):
        x, y = sampler.sample_batch(it1, id, args.minibatch_size)
        x_qry.append(x)
        y_qry.append(y)

    x_qry = torch.stack([torch.cat(x_qry)])
    y_qry = torch.stack([torch.cat(y_qry)])

    rand_indices = list(range(len(x_spt)))
    np.random.shuffle(rand_indices)
    if iid:
        x_spt_new = []
        y_spt_new = []
        for a in rand_indices:
            x_spt_new.append(x_spt[a])
            y_spt_new.append(y_spt[a])
        x_spt = x_spt_new
        y_spt = y_spt_new

    x_spt = torch.stack(x_spt)
    y_spt = torch.stack(y_spt)

    return x_spt, y_spt, x_qry, y_qry


def main(args):
    # Seed random number generators
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    my_experiment = experiment(args.name, args, "/data5/jlindsey/continual/results", commit_changes=args.commit)
    writer = SummaryWriter(my_experiment.path + "tensorboard")
    print(args)

    # Initalize tasks; we sample 1000 tasks for evaluation
    tasks = list(range(1000))
    logger = logging.getLogger('experiment')

    sampler = ts.SamplerFactory.get_sampler("Sin", tasks, None, None, capacity=args.capacity + 1)

    #config = mf.ModelFactory.get_model("na", "Sin", in_channels=args.capacity + 1, num_actions=args.tasks)

    config = mf.ModelFactory.get_model(args.modeltype, "Sin", in_channels=args.capacity + 1, num_actions=1,
                                       width=args.width)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load the model
    #print('config is', config)
    maml = MetaLearnerRegression(args, config).to(device)
    maml.net = torch.load(args.model, map_location='cpu').to(device)

    for name, param in maml.named_parameters():
        if name.find("feedback_strength_vars") != -1:
            print(name, param)
        param.learn = True
    for name, param in maml.net.named_parameters():
        param.learn = True

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    logger.info(maml)
    logger.info('Total trainable tensors: %d', num)

    ##### Setting up parameters for freezing RLN layers
    #### Also resets TLN layers with random initialization if args.reset is true
    frozen_layers = []
    for temp in range(args.rln * 2):
        frozen_layers.append("net.vars." + str(temp))

    for name, param in maml.named_parameters():
        logger.info(name)
        if name in frozen_layers:
            logger.info("Freeezing name %s", str(name))
            param.learn = False
            logger.info(str(param.requires_grad))
        else:
            if args.reset:
                w = nn.Parameter(torch.ones_like(param))
                if len(w.shape) > 1:
                    logger.info("Resseting layer %s", str(name))
                    torch.nn.init.kaiming_normal_(w)
                else:
                    w = nn.Parameter(torch.zeros_like(param))
                param.data = w
                param.learn = True

    for name, param in maml.net.named_parameters():
        logger.info(name)
        if name in frozen_layers:
            logger.info("Freeezing name %s", str(name))
            param.learn = False
            logger.info(str(param.requires_grad))

    correct = 0
    counter = 0
    for name, _ in maml.net.named_parameters():
        # logger.info("LRs of layer %s = %s", str(name), str(torch.mean(maml.lrs[counter])))
        counter += 1

    for lrs in [0]:
        loss_vector = np.zeros(args.tasks)
        loss_vector_results = []
        lr_results = {}
        incremental_results = {}
        lr_results[lrs] = []

        runs = args.runs
        for temp in range(0, runs):
            loss_vector = np.zeros(args.tasks)
            t1 = np.random.choice(tasks, args.tasks, replace=False)
            print(temp, t1)

            iterators = []
            for t in t1:
                iterators.append(sampler.sample_task([t]))
            if args.vary_length:
                num_steps = np.random.randint(args.update_step//10, args.update_step+1)
                x_spt, y_spt, x_qry, y_qry = construct_set(iterators, sampler, steps=num_steps, iid=args.iid)
            else:
                num_steps = args.update_step
                x_spt, y_spt, x_qry, y_qry = construct_set(iterators, sampler, steps=args.update_step, iid=args.iid)
            if torch.cuda.is_available():
                x_spt, y_spt, x_qry, y_qry = x_spt.cuda(), y_spt.cuda(), x_qry.cuda(), y_qry.cuda()

                
            #print("TESTING", len(x_spt), args.update_step)
            net = copy.deepcopy(maml.net)
            net = net.to(device)
            for params_old, params_new in zip(maml.net.parameters(), net.parameters()):
                params_new.learn = params_old.learn

            list_of_params = list(filter(lambda x: x.learn, net.parameters()))

            #optimizer = optim.SGD(list_of_params, lr=lrs)

            counter = 0
            x_spt_test, y_spt_test, x_qry_test, y_qry_test = construct_set(iterators, sampler, steps=300)
            if args.train_performance:
                x_spt_test, y_spt_test, x_qry_test, y_qry_test = x_spt, y_spt, x_qry, y_qry
                x_qry_test, y_qry_test = x_spt_test, y_spt_test
            if torch.cuda.is_available():
                x_spt_test, y_spt_test, x_qry_test, y_qry_test = x_spt_test.cuda(), y_spt_test.cuda(), x_qry_test.cuda(), y_qry_test.cuda()
            fast_weights = None
            if args.randomize_plastic_weights:
                net.randomize_plastic_weights()
            if args.zero_plastic_weights:
                net.zero_plastic_weights()
            for k in range(len(x_spt)):
                if k % num_steps == 0 and k > 0:
                    counter += 1
                    loss_temp = 0
                    if not counter in incremental_results:
                        incremental_results[counter] = []
                    with torch.no_grad():
                        if args.train_performance:
                            for update_upto in range(0, k):
                                logits = net(x_spt_test[update_upto], vars=fast_weights, bn_training=False)

                                logits_select = []
                                for no, val in enumerate(y_spt_test[update_upto, :, 1].long()):
                                    logits_select.append(logits[no, val])
                                logits = torch.stack(logits_select).unsqueeze(1)
                                loss_temp += F.mse_loss(logits, y_spt_test[update_upto, :, 0].unsqueeze(1))

                            loss_temp = loss_temp / (k)
                        else:
                            for update_upto in range(0, counter * 300):
                                logits = net(x_spt_test[update_upto], vars=fast_weights, bn_training=False)

                                logits_select = []
                                for no, val in enumerate(y_spt_test[update_upto, :, 1].long()):
                                    logits_select.append(logits[no, val])
                                logits = torch.stack(logits_select).unsqueeze(1)
                                loss_temp += F.mse_loss(logits, y_spt_test[update_upto, :, 0].unsqueeze(1))

                            loss_temp = loss_temp / (counter * 300)
                        incremental_results[counter].append(loss_temp.item())
                        my_experiment.results["incremental"] = incremental_results

                logits = net(x_spt[k], fast_weights, bn_training=False)

                logits_select = []
                for no, val in enumerate(y_spt[k, :, 1].long()):
                    logits_select.append(logits[no, val])

                logits = torch.stack(logits_select).unsqueeze(1)
                loss = F.mse_loss(logits, y_spt[k, :, 0].unsqueeze(1))
                fast_weights = net.getOjaUpdate(y_spt[k, :, 0:1], logits, fast_weights, hebbian=maml.hebb)
                #optimizer.zero_grad()
                #loss.backward()
                #optimizer.step()

            counter += 1
            loss_temp = 0
            if not counter in incremental_results:
                incremental_results[counter] = []
            with torch.no_grad():
                if args.train_performance:
                    for update_upto in range(0, k):
                        logits = net(x_spt_test[update_upto], vars=fast_weights, bn_training=False)

                        logits_select = []
                        for no, val in enumerate(y_spt_test[update_upto, :, 1].long()):
                            logits_select.append(logits[no, val])
                        logits = torch.stack(logits_select).unsqueeze(1)
                        loss_temp += F.mse_loss(logits, y_spt_test[update_upto, :, 0].unsqueeze(1))
                        # lr_results[lrs].append(loss_q.item())
                    loss_temp = loss_temp / (k)
                else:
                    for update_upto in range(0, counter * 300):
                        logits = net(x_spt_test[update_upto], vars=fast_weights, bn_training=False)

                        logits_select = []
                        for no, val in enumerate(y_spt_test[update_upto, :, 1].long()):
                            logits_select.append(logits[no, val])
                        logits = torch.stack(logits_select).unsqueeze(1)
                        loss_temp += F.mse_loss(logits, y_spt_test[update_upto, :, 0].unsqueeze(1))
                        # lr_results[lrs].append(loss_q.item())
                    loss_temp = loss_temp / (counter * 300)
                incremental_results[counter].append(loss_temp.item())
                my_experiment.results["incremental"] = incremental_results
            #
            x_spt, y_spt, x_qry, y_qry = x_spt_test, y_spt_test, x_qry_test, y_qry_test
            if torch.cuda.is_available():
                x_spt, y_spt, x_qry, y_qry = x_spt.cuda(), y_spt.cuda(), x_qry.cuda(), y_qry.cuda()
            with torch.no_grad():
                logits = net(x_qry[0], vars=fast_weights, bn_training=False)

                logits_select = []
                for no, val in enumerate(y_qry[0, :, 1].long()):
                    logits_select.append(logits[no, val])
                logits = torch.stack(logits_select).unsqueeze(1)
                loss_q = F.mse_loss(logits, y_qry[0, :, 0].unsqueeze(1))
                lr_results[lrs].append(loss_q.item())

            counter = 0
            loss = 0

            for k in range(len(x_spt)):

                logits = net(x_spt[k], vars=fast_weights, bn_training=False)

                logits_select = []
                for no, val in enumerate(y_spt[k, :, 1].long()):
                    logits_select.append(logits[no, val])
                logits = torch.stack(logits_select).unsqueeze(1)

                loss_vector[int(counter / (300))] += F.mse_loss(logits, y_spt[k, :, 0].unsqueeze(1)) / 300

                counter += 1
            loss_vector_results.append(loss_vector.tolist())
            print('avg loss', np.mean(lr_results[lrs]))

        logger.info("Loss vector all %s", str(loss_vector_results))
        logger.info("Avg MSE LOSS  for lr %s = %s", str(lrs), str(np.mean(lr_results[lrs])))
        logger.info("Std MSE LOSS  for lr %s = %s", str(lrs), str(np.std(lr_results[lrs])))
        loss_vector = loss_vector / runs
        print("Loss vector = ", loss_vector)
        my_experiment.results[str(lrs)] = str(loss_vector_results)
        my_experiment.store_json()
        np.save('evals/loss_vector_results_'+args.orig_name+'.npy', loss_vector_results) 
        np.save('evals/final_results_'+args.orig_name+'.npy', lr_results) 
        np.save('evals/incremental_results_'+args.orig_name+'.npy', incremental_results) 
        
        print('lv results', loss_vector_results)
        print('final_results', lr_results)
        print('incremental_results', incremental_results)
    #torch.save(maml.net, my_experiment.path + "learner.model")


# #

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=20000)
    argparser.add_argument('--seed', type=int, help='Seed for random', default=103450)
    argparser.add_argument('--seeds', type=int, nargs='+', help='n way', default=[10])
    argparser.add_argument('--classes', type=int, nargs='+', help='Total classes to use in training',
                           default=[0, 1, 2, 3, 4])
    argparser.add_argument('--model', type=str, help='epoch number', default="none")
    argparser.add_argument('--tasks', type=int, help='meta batch size, namely task num', default=10)
    argparser.add_argument('--capacity', type=int, help='meta batch size, namely task num', default=10)
    argparser.add_argument('--runs', type=int, help='meta batch size, namely task num', default=50)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-4)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.003)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=40)
    argparser.add_argument('--name', help='Name of experiment', default="dolphin")
    argparser.add_argument('--reset', action="store_true")
    argparser.add_argument("--commit", action="store_true")
    argparser.add_argument("--iid", action="store_true")
    argparser.add_argument("--no-freeze", action="store_true")
    argparser.add_argument("--rln", type=int, default=6)
    argparser.add_argument("--hebb", action="store_true") #don't use if --oja is set
    argparser.add_argument('--feedback_strength', type=float, help='initial value for how much the feedback affects the activation.', default=0.5)
    #argparser.add_argument("--trainable_plasticity", action="store_true")
    argparser.add_argument('--init_plasticity', type=float, help='initial plasticity rate', default=0.001)
    #argparser.add_argument("--optimize_feedback", action="store_true")
    argparser.add_argument("--train_on_new", action="store_true")
    #argparser.add_argument("--propagate_feeedback", action="store_true")
    argparser.add_argument('--num_extra_dense_layers', type=int, help='num dense layers in addition to one intermediate layer', default=0)
    argparser.add_argument('--num_feedback_layers', type=int, help='num dense layers in feedback', default=1)
    #argparser.add_argument('--num_extra_nonplastic_dense_output_layers', type=int, help='num nonplastic linear layers in addition to one output layer', default=0)
    argparser.add_argument("--rln_end", type=int, default=0)
    argparser.add_argument("--no_class_reset", action="store_true")
    argparser.add_argument("--all_class_reset", action="store_true")
    argparser.add_argument("--zero_non_output_plasticity", action="store_true")
    argparser.add_argument("--zero_all_plasticity", action="store_true")
    argparser.add_argument("--feedback_l2", type=int, default=0.0)
    argparser.add_argument("--overwrite", action="store_true")
    argparser.add_argument("--optimize_out", action="store_true")
    argparser.add_argument("--plasticity_rank1", action="store_true")
    argparser.add_argument("--freeze_out_plasticity", action="store_true")
    argparser.add_argument("--simul_feedback", action="store_true")
    argparser.add_argument("--use_error", action="store_true")
    
    argparser.add_argument('--meta_feedback_lr', type=float, help='meta-level outer learning rate for feedback weights', default=1e-4)
    argparser.add_argument('--meta_plasticity_lr', type=float, help='meta-level outer learning rate for plasticity', default=1e-4)
    argparser.add_argument('--meta_feedback_strength_lr', type=float, help='meta-level outer learning rate for feedback strength', default=1e-4) 

    argparser.add_argument("--width", type=int, default=300)
    argparser.add_argument('--modeltype', help='Name of model', default="old")
    argparser.add_argument("--randomize_plastic_weights", action="store_true")
    argparser.add_argument("--zero_plastic_weights", action="store_true")
    argparser.add_argument("--train_performance", action="store_true")
    argparser.add_argument('--minibatch_size', type=int, help='epoch number', default=32)
    argparser.add_argument("--feedback_only_to_output", action="store_true")
    argparser.add_argument("--linear_feedback", action="store_true")
    argparser.add_argument("--use_derivative", action="store_true")
    argparser.add_argument("--error_only_to_output", action="store_true")
    argparser.add_argument("--neuron_level_plasticity", action="store_true")
    argparser.add_argument("--layer_level_plasticity", action="store_true")
    argparser.add_argument("--coarse_level_plasticity", action="store_true")
    argparser.add_argument("--vary_length", action="store_true")





    args = argparser.parse_args()

    args.orig_name = args.name
    args.name = "/".join(["sin", "evaluate", args.name])
    print(args)
    main(args)

