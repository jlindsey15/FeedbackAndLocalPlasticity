import argparse
import copy
import logging

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import optim
from torch.nn import functional as F

import datasets.task_sampler as ts
import model.modelfactory as mf
from experiment.experiment import experiment
from model.meta_learner import MetaLearnerRegression
from model.oja_meta_learner import MetaLearnerRegression as OjaMetaLearnerRegression


logger = logging.getLogger('experiment')


def construct_set(iterators, sampler, steps):
    x_traj = []
    y_traj = []
    list_of_ids = list(range(sampler.capacity - 1))

    start_index = 0

    for id, it1 in enumerate(iterators):
        for inner in range(steps):
            x, y = sampler.sample_batch(it1, list_of_ids[(id + start_index) % len(list_of_ids)], args.minibatch_size)
            x_traj.append(x)
            y_traj.append(y)
    #

    x_rand = []
    y_rand = []
    for id, it1 in enumerate(iterators):
        x, y = sampler.sample_batch(it1, list_of_ids[(id + start_index) % len(list_of_ids)], args.minibatch_size)
        x_rand.append(x)
        y_rand.append(y)

    x_rand = torch.stack([torch.cat(x_rand)])
    y_rand = torch.stack([torch.cat(y_rand)])

    x_traj = torch.stack(x_traj)
    y_traj = torch.stack(y_traj)

    return x_traj, y_traj, x_rand, y_rand

def construct_set_iid(iterators, sampler, steps=2, iid=True):
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
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    my_experiment = experiment(args.name, args, "/data5/jlindsey/continual/results", commit_changes=args.commit)
    writer = SummaryWriter(my_experiment.path + "tensorboard")
    print(args)

    tasks = list(range(400))
    logger = logging.getLogger('experiment')

    sampler = ts.SamplerFactory.get_sampler("Sin", tasks, None, capacity=args.capacity + 1)

    config = mf.ModelFactory.get_model(args.modeltype, "Sin", in_channels=args.capacity + 1, num_actions=1,
                                       width=args.width)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.oja or args.hebb:
        maml = OjaMetaLearnerRegression(args, config).to(device)
    else:
        maml = MetaLearnerRegression(args, config).to(device)
        old_vars_plasticity = maml.net.vars_plasticity
        old_coarse_plasticity = maml.net.coarse_plasticity
        old_layer_level_vars_plasticity = maml.net.layer_level_vars_plasticity
        
    old_means = []
    old_stds = []
    for param in maml.net.vars:
        print('mean std old', param.cpu().detach().numpy().mean(), param.cpu().detach().numpy().std())
        old_means.append(torch.mean(param))
        old_stds.append(torch.std(param))
    if args.reset_non_feedforward:
        old_feedback_vars = maml.net.feedback_vars
        old_feedback_vars_bundled = maml.net.feedback_vars_bundled
        old_plasticity = maml.net.plasticity
        old_neuron_plasticity = maml.net.neuron_plasticity
        old_layer_plasticity = maml.net.layer_plasticity
        old_vars_bn = maml.net.vars_bn
        
    
    if args.from_saved:
        maml.net = torch.load(args.model)
        if ((not (args.oja or args.hebb)) and (not hasattr(maml.net, 'vars_plasticity'))):
            maml.net.vars_plasticity = old_vars_plasticity
        if ((not (args.oja or args.hebb)) and (not hasattr(maml.net, 'coarse_plasticity'))):
            maml.net.coarse_plasticity =  old_coarse_plasticity
        if ((not (args.oja or args.hebb)) and (not hasattr(maml.net, 'layer_level_vars_plasticity'))):
            maml.net.layer_level_vars_plasticity =  old_layer_level_vars_plasticity
        maml.net.optimize_out = args.optimize_out
        if maml.net.optimize_out:
            maml.net.feedback_strength_vars.append(torch.nn.Parameter(maml.net.init_feedback_strength * torch.ones(1).cuda()))
        if args.reset_non_feedforward:
            maml.net.feedback_vars = old_feedback_vars
            maml.net.feedback_vars_bundled = old_feedback_vars_bundled
            maml.net.plasticity = old_plasticity
            maml.net.neuron_plasticity = old_neuron_plasticity
            maml.net.layer_plasticity = old_layer_plasticity
            maml.net.vars_bn = old_vars_bn
        maml.init_stuff(args)
        

    
    
    if args.scale_from_load:
        template = torch.load(args.model)
        for var_idx in range(len(maml.net.vars)):
            maml.net.vars[var_idx] = torch.nn.Parameter(maml.net.vars[var_idx] - torch.mean(maml.net.vars[var_idx]))
            
            if var_idx < len(maml.net.vars) - 1:
                maml.net.vars[var_idx] = torch.nn.Parameter(maml.net.vars[var_idx] / (1e-8+torch.std(maml.net.vars[var_idx])))
                maml.net.vars[var_idx] = torch.nn.Parameter(maml.net.vars[var_idx] * torch.std(template.vars[var_idx]))
            
            maml.net.vars[var_idx] = torch.nn.Parameter(maml.net.vars[var_idx] + torch.mean(template.vars[var_idx]))
        
    
    for param in maml.net.vars:
        print('mean std new', param.cpu().detach().numpy().mean(), param.cpu().detach().numpy().std())
    maml.net.optimize_out = args.optimize_out
    if maml.net.optimize_out:
        maml.net.feedback_strength_vars.append(torch.nn.Parameter(maml.net.init_feedback_strength * torch.ones(1).cuda()))
    #I recently un-indented this until the maml.init_opt() line.  If stuff stops working, try re-indenting this block
    if args.zero_non_output_plasticity:
        for index in range(len(maml.net.plasticity) - 1):
            if args.plasticity_rank1:
                maml.net.plasticity[index] = torch.nn.Parameter(torch.zeros(1).cuda())
            else:
                maml.net.plasticity[index] = torch.nn.Parameter(maml.net.plasticity[index] * 0)
    if args.zero_all_plasticity:
        for index in range(len(maml.net.plasticity)):
            if args.plasticity_rank1:
                maml.net.plasticity[index] = torch.nn.Parameter(torch.zeros(1).cuda())
            else:
                maml.net.plasticity[index] = torch.nn.Parameter(maml.net.plasticity[index] * 0)

    maml.init_opt()

    for name, param in maml.named_parameters():
        param.learn = True
    for name, param in maml.net.named_parameters():
        param.learn = True
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    logger.info(maml)
    logger.info('Total trainable tensors: %d', num)
    #
    accuracy = 0

    frozen_layers = []
    if args.freeze_out_plasticity:
        maml.net.plasticity[-1].requires_grad = False
    total_ff_vars = 2*(7 + 2 + args.num_extra_dense_layers)
    frozen_layers = []
    for temp in range(args.rln * 2):
        frozen_layers.append("net.vars." + str(temp))

    for temp in range(args.rln_end * 2):
        frozen_layers.append("net.vars." + str(total_ff_vars - 1 - temp))
        
     
    logger.info("Frozen layers = %s", " ".join(frozen_layers))
    for step in range(args.epoch):

        if step == 0:
            for name, param in maml.named_parameters():
                logger.info(name)
                if name in frozen_layers:
                    logger.info("Freeezing name %s", str(name))
                    param.learn = False
                    logger.info(str(param.requires_grad))

            for name, param in maml.net.named_parameters():
                logger.info(name)
                if name in frozen_layers:
                    logger.info("Freeezing name %s", str(name))
                    param.learn = False
                    logger.info(str(param.requires_grad))

        t1 = np.random.choice(tasks, args.tasks, replace=False)

        iterators = []
        for t in t1:
            # print(sampler.sample_task([t]))
            iterators.append(sampler.sample_task([t]))

        if args.iid:
            if args.vary_length:
                x_traj, y_traj, x_rand, y_rand = construct_set_iid(iterators, sampler, steps=np.random.randint(args.update_step//10, args.update_step+1))                
            else:
                x_traj, y_traj, x_rand, y_rand = construct_set_iid(iterators, sampler, steps=args.update_step)
        else:
            if args.vary_length:
                x_traj, y_traj, x_rand, y_rand = construct_set(iterators, sampler, steps=np.random.randint(args.update_step//10, args.update_step+1))                
            else:
                x_traj, y_traj, x_rand, y_rand = construct_set(iterators, sampler, steps=args.update_step)

        #print("TESTING", len(x_traj), args.update_step)
        if torch.cuda.is_available():
            x_traj, y_traj, x_rand, y_rand = x_traj.cuda(), y_traj.cuda(), x_rand.cuda(), y_rand.cuda()
        # print(x_spt, y_spt)
        accs = maml(x_traj, y_traj, x_rand, y_rand)
        #print('accs', accs)
        maml.meta_optim.step()

        if step in [0, 2000, 3000, 4000]:
            for param_group in maml.optimizer.param_groups:
                logger.info("Learning Rate at step %d = %s", step, str(param_group['lr']))

        accuracy = accuracy * 0.95 + 0.05 * accs[-1]
        if step % 5 == 0:
            writer.add_scalar('/metatrain/train/accuracy', accs[-1], step)
            writer.add_scalar('/metatrain/train/runningaccuracy', accuracy, step)
            logger.info("Running average of accuracy = %s", str(accuracy))
            logger.info('step: %d \t training acc (first, last) %s', step, str(accs[0]) + "," + str(accs[-1]))

        if step % 100 == 0:
            counter = 0
            for name, _ in maml.net.named_parameters():
                counter += 1
            '''
            for lrs in [args.update_lr]:
                lr_results = {}
                lr_results[lrs] = []
                for temp in range(0, 20):
                    t1 = np.random.choice(tasks, args.tasks, replace=False)
                    iterators = []

                    for t in t1:
                        iterators.append(sampler.sample_task([t]))
                    x_traj, y_traj, x_rand, y_rand = construct_set(iterators, sampler, steps=40)
                    if torch.cuda.is_available():
                        x_traj, y_traj, x_rand, y_rand = x_traj.cuda(), y_traj.cuda(), x_rand.cuda(), y_rand.cuda()

                    net = copy.deepcopy(maml.net)
                    net = net.to(device)
                    for params_old, params_new in zip(maml.net.parameters(), net.parameters()):
                        params_new.learn = params_old.learn

                    list_of_params = list(filter(lambda x: x.learn, net.parameters()))

                    optimizer = optim.SGD(list_of_params, lr=lrs)
                    for k in range(len(x_traj)):
                        logits = net(x_traj[k], None, bn_training=False)

                        logits_select = []
                        for no, val in enumerate(y_traj[k, :, 1].long()):
                            logits_select.append(logits[no, val])

                        logits = torch.stack(logits_select).unsqueeze(1)

                        loss = F.mse_loss(logits, y_traj[k, :, 0].unsqueeze(1))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    #
                    with torch.no_grad():
                        logits = net(x_rand[0], vars=None, bn_training=False)

                        logits_select = []
                        for no, val in enumerate(y_rand[0, :, 1].long()):
                            logits_select.append(logits[no, val])
                        logits = torch.stack(logits_select).unsqueeze(1)
                        loss_q = F.mse_loss(logits, y_rand[0, :, 0].unsqueeze(1))
                        lr_results[lrs].append(loss_q.item())

                logger.info("Avg MSE LOSS  for lr %s = %s", str(lrs), str(np.mean(lr_results[lrs])))
            '''
            torch.save(maml.net, my_experiment.path + "learner.model")
    torch.save(maml.net, my_experiment.path + "learner.model")


#

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=20000)
    argparser.add_argument('--seed', type=int, help='Seed for random', default=1000)
    argparser.add_argument('--tasks', type=int, help='meta batch size, namely task num', default=10)
    argparser.add_argument('--capacity', type=int, help='meta batch size, namely task num', default=10)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-4)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.003)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=40)
    argparser.add_argument('--name', help='Name of experiment', default="mrcl_regression")
    argparser.add_argument('--modeltype', help='Name of model', default="old")
    argparser.add_argument("--commit", action="store_true")
    argparser.add_argument("--width", type=int, default=300)
    argparser.add_argument("--rln", type=int, default=6)
    
    
    argparser.add_argument('--meta_feedback_lr', type=float, help='meta-level outer learning rate for feedback weights', default=1e-4)
    argparser.add_argument('--meta_plasticity_lr', type=float, help='meta-level outer learning rate for plasticity', default=1e-4)
    argparser.add_argument('--meta_feedback_strength_lr', type=float, help='meta-level outer learning rate for feedback strength', default=1e-4)

    argparser.add_argument("--oja", action="store_true") #don't use if --hebb is set
    argparser.add_argument("--hebb", action="store_true") #don't use if --oja is set
    #argparser.add_argument("--feedback_strength_clamp", action="store_true") #don't use if --oja is set
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
    argparser.add_argument("--from_saved", action="store_true")
    argparser.add_argument("--zero_non_output_plasticity", action="store_true")
    argparser.add_argument("--zero_all_plasticity", action="store_true")
    argparser.add_argument("--model", type=str, default='')
    argparser.add_argument("--feedback_l2", type=int, default=0.0)
    argparser.add_argument("--overwrite", action="store_true")
    argparser.add_argument("--optimize_out", action="store_true")
    argparser.add_argument("--plasticity_rank1", action="store_true")
    argparser.add_argument("--freeze_out_plasticity", action="store_true")
    argparser.add_argument("--simul_feedback", action="store_true")
    argparser.add_argument("--use_error", action="store_true")
    argparser.add_argument("--plastic_update", action="store_true")
    argparser.add_argument("--layer_level_plastic_update", action="store_true")


    argparser.add_argument("--coarse_plastic_update", action="store_true")


    argparser.add_argument("--randomize_plastic_weights", action="store_true")
    argparser.add_argument("--zero_plastic_weights", action="store_true")
    argparser.add_argument("--iid", action="store_true")
    argparser.add_argument('--minibatch_size', type=int, help='epoch number', default=32)
    argparser.add_argument("--feedback_only_to_output", action="store_true")
    argparser.add_argument("--error_only_to_output", action="store_true")
    argparser.add_argument("--linear_feedback", action="store_true")
    argparser.add_argument("--use_derivative", action="store_true")
    argparser.add_argument("--neuron_level_plasticity", action="store_true")
    argparser.add_argument("--layer_level_plasticity", action="store_true")
    argparser.add_argument("--coarse_level_plasticity", action="store_true")


    argparser.add_argument("--vary_length", action="store_true")
    argparser.add_argument("--reset_non_feedforward", action="store_true")
    argparser.add_argument("--scale_from_load", action="store_true")






    
    args = argparser.parse_args()

    args.name = args.name#"/".join([args.dataset, str(args.meta_lr).replace(".", "_"), args.name])
    print(args)
    main(args)


