import argparse
import logging

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch import nn

import datasets.datasetfactory as df
import datasets.task_sampler as ts
import model.modelfactory as mf
from experiment.experiment import experiment
from model.meta_learner import MetaLearingClassification
from model.oja_meta_learner import MetaLearingClassification as OjaMetaLearingClassification
import datasets.miniimagenet as imgnet


logger = logging.getLogger('experiment')


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    my_experiment = experiment(args.name, args, "/data5/jlindsey/continual/results", commit_changes=args.commit)
    writer = SummaryWriter(my_experiment.path + "tensorboard")

    logger = logging.getLogger('experiment')

    args.classes = list(range(963))
    
    
    print('dataset', args.dataset, args.dataset == "imagenet")

    if args.dataset != "imagenet":
        
        dataset = df.DatasetFactory.get_dataset(args.dataset, background=True, train=True, all=True)
        dataset_test = df.DatasetFactory.get_dataset(args.dataset, background=True, train=False, all=True)

    else:
        args.classes = list(range(64))
        dataset = imgnet.MiniImagenet(args.imagenet_path, mode='train')
        dataset_test = imgnet.MiniImagenet(args.imagenet_path, mode='test')
        
        
    iterator_test = torch.utils.data.DataLoader(dataset_test, batch_size=5,
                                                shuffle=True, num_workers=1)

    iterator_train = torch.utils.data.DataLoader(dataset, batch_size=5,
                                                 shuffle=True, num_workers=1)

    logger.info("Train set length = %d", len(iterator_train) * 5)
    logger.info("Test set length = %d", len(iterator_test) * 5)
    sampler = ts.SamplerFactory.get_sampler(args.dataset, args.classes, dataset, dataset_test)

    config = mf.ModelFactory.get_model(args.model_type, args.dataset, width=args.width, num_extra_dense_layers=args.num_extra_dense_layers)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')



    if args.oja or args.hebb:
      maml = OjaMetaLearingClassification(args, config).to(device)
    else:
      print('starting up')
      maml = MetaLearingClassification(args, config).to(device)
        
    
    import sys
    if args.from_saved:
        maml.net = torch.load(args.model)
        if args.use_derivative:
            maml.net.use_derivative = True
        maml.net.optimize_out = args.optimize_out
        if maml.net.optimize_out:
            maml.net.feedback_strength_vars.append(torch.nn.Parameter(maml.net.init_feedback_strength * torch.ones(1).cuda()))
        
            
        if args.reset_feedback_strength:
            for fv in maml.net.feedback_strength_vars:
                w = nn.Parameter(torch.ones_like(fv)*args.feedback_strength)
                fv.data = w   
                
        if args.reset_feedback_vars:
            
            print('howdy', maml.net.num_feedback_layers)

            maml.net.feedback_vars = nn.ParameterList()
            maml.net.feedback_vars_bundled = []
            
            maml.net.vars_plasticity = nn.ParameterList()
            maml.net.plasticity = nn.ParameterList()


            maml.net.neuron_plasticity = nn.ParameterList()

            maml.net.layer_plasticity = nn.ParameterList()


            
            starting_width = 84
            cur_width = starting_width
            num_outputs = maml.net.config[-1][1][0]
            for i, (name, param) in enumerate(maml.net.config):
                print('yo', i, name, param)
                if name == 'conv2d':
                    print('in conv2d')
                    stride=param[4]
                    padding=param[5]

                    #print('cur_width', cur_width, param[3])
                    cur_width = (cur_width + 2*padding - param[3] + stride) // stride

                    maml.net.vars_plasticity.append(nn.Parameter(torch.ones(*param[:4]).cuda()))
                    maml.net.vars_plasticity.append(nn.Parameter(torch.ones(param[0]).cuda()))
                    #self.activations_list.append([])
                    maml.net.plasticity.append(nn.Parameter(maml.net.init_plasticity * torch.ones(param[0], param[1]*param[2]*param[3]).cuda())) #not implemented
                    maml.net.neuron_plasticity.append(nn.Parameter(torch.zeros(1).cuda())) #not implemented

                    maml.net.layer_plasticity.append(nn.Parameter(maml.net.init_plasticity * torch.ones(1).cuda())) #not implemented
                
                    feedback_var = []
                    
                    
                    for fl in range(maml.net.num_feedback_layers):
                        print('doing fl')
                        in_dim = maml.net.width
                        out_dim = maml.net.width
                        if fl == maml.net.num_feedback_layers - 1:
                            out_dim = param[0] * cur_width * cur_width
                        if fl == 0:
                            in_dim = num_outputs
                        feedback_w_shape = [out_dim, in_dim]
                        feedback_w = nn.Parameter(torch.ones(feedback_w_shape).cuda())
                        feedback_b =  nn.Parameter(torch.zeros(out_dim).cuda())
                        torch.nn.init.kaiming_normal_(feedback_w)
                        feedback_var.append((feedback_w, feedback_b))
                        print('adding')
                        maml.net.feedback_vars.append(feedback_w)
                        maml.net.feedback_vars.append(feedback_b)


                    #maml.net.feedback_vars_bundled.append(feedback_var)
                    #maml.net.feedback_vars_bundled.append(None)#bias feedback -- not implemented

                    #'''

                    maml.net.feedback_vars_bundled.append(nn.Parameter(torch.zeros(1)))#weight feedback -- not implemented
                    maml.net.feedback_vars_bundled.append(nn.Parameter(torch.zeros(1)))#bias feedback -- not implemented


                elif name == 'linear':
                    maml.net.vars_plasticity.append(nn.Parameter(torch.ones(*param).cuda()))
                    maml.net.vars_plasticity.append(nn.Parameter(torch.ones(param[0]).cuda()))
                    #self.activations_list.append([])
                    maml.net.plasticity.append(nn.Parameter(maml.net.init_plasticity * torch.ones(*param).cuda()))
                    maml.net.neuron_plasticity.append(nn.Parameter(maml.net.init_plasticity * torch.ones(param[0]).cuda()))
                    maml.net.layer_plasticity.append(nn.Parameter(maml.net.init_plasticity * torch.ones(1).cuda()))


                    feedback_var = []

                    for fl in range(maml.net.num_feedback_layers):
                        in_dim = maml.net.width
                        out_dim = maml.net.width
                        if fl == maml.net.num_feedback_layers - 1:
                            out_dim = param[0]
                        if fl == 0:
                            in_dim = num_outputs
                        feedback_w_shape = [out_dim, in_dim]
                        feedback_w = nn.Parameter(torch.ones(feedback_w_shape).cuda())
                        feedback_b =  nn.Parameter(torch.zeros(out_dim).cuda())
                        torch.nn.init.kaiming_normal_(feedback_w)
                        feedback_var.append((feedback_w, feedback_b))
                        maml.net.feedback_vars.append(feedback_w)
                        maml.net.feedback_vars.append(feedback_b)
                    maml.net.feedback_vars_bundled.append(feedback_var)
                    maml.net.feedback_vars_bundled.append(None)#bias feedback -- not implemented


                
        maml.init_stuff(args)
     
    maml.net.optimize_out = args.optimize_out
    if maml.net.optimize_out:
        maml.net.feedback_strength_vars.append(torch.nn.Parameter(maml.net.init_feedback_strength * torch.ones(1).cuda()))
    #I recently un-indented this until the maml.init_opt() line.  If stuff stops working, try re-indenting this block
    if args.zero_non_output_plasticity:
        for index in range(len(maml.net.vars_plasticity)-2):
            maml.net.vars_plasticity[index] = torch.nn.Parameter(maml.net.vars_plasticity[index] * 0)
        if args.oja or args.hebb:
            for index in range(len(maml.net.plasticity) - 1):
                if args.plasticity_rank1:
                    maml.net.plasticity[index] = torch.nn.Parameter(torch.zeros(1).cuda())
                else:
                    maml.net.plasticity[index] = torch.nn.Parameter(maml.net.plasticity[index] * 0)
                    maml.net.layer_plasticity[index] = torch.nn.Parameter(maml.net.layer_plasticity[index] * 0)
                    maml.net.neuron_plasticity[index] = torch.nn.Parameter(maml.net.neuron_plasticity[index] * 0)
                    
        if args.oja or args.hebb:
            for index in range(len(maml.net.vars_plasticity) - 2):
                maml.net.vars_plasticity[index] = torch.nn.Parameter(maml.net.vars_plasticity[index] * 0)
    if args.zero_all_plasticity:
        print('zeroing plasticity')
        for index in range(len(maml.net.vars_plasticity)):
            maml.net.vars_plasticity[index] = torch.nn.Parameter(maml.net.vars_plasticity[index] * 0)
        for index in range(len(maml.net.plasticity)):
            if args.plasticity_rank1:
                maml.net.plasticity[index] = torch.nn.Parameter(torch.zeros(1).cuda())
            else:
                
                maml.net.plasticity[index] = torch.nn.Parameter(maml.net.plasticity[index] * 0)
                maml.net.layer_plasticity[index] = torch.nn.Parameter(maml.net.layer_plasticity[index] * 0)
                maml.net.neuron_plasticity[index] = torch.nn.Parameter(maml.net.neuron_plasticity[index] * 0)
                
    print('heyy', maml.net.feedback_vars)
    maml.init_opt()
    for name, param in maml.named_parameters():
        param.learn = True
    for name, param in maml.net.named_parameters():
        param.learn = True

    if args.freeze_out_plasticity:
        maml.net.plasticity[-1].requires_grad = False
    total_ff_vars = 2*(6 + 2 + args.num_extra_dense_layers)
    frozen_layers = []
    for temp in range(args.rln * 2):
        frozen_layers.append("net.vars." + str(temp))

    for temp in range(args.rln_end * 2):
        frozen_layers.append("net.vars." + str(total_ff_vars - 1 - temp))
    for name, param in maml.named_parameters():
        # logger.info(name)
        if name in frozen_layers:
            logger.info("RLN layer %s", str(name))
            param.learn = False

    # Update the classifier
    list_of_params = list(filter(lambda x: x.learn, maml.parameters()))
    list_of_names = list(filter(lambda x: x[1].learn, maml.named_parameters()))

    for a in list_of_names:
        logger.info("TLN layer = %s", a[0])

    for step in range(args.steps):
        '''
        print('plasticity')
        for p in maml.net.plasticity:
            print(p.size(), torch.sum(p), p)
        '''
        t1 = np.random.choice(args.classes, args.tasks, replace=False)#np.random.randint(1, args.tasks + 1), replace=False)

        d_traj_iterators = []
        for t in t1:
            d_traj_iterators.append(sampler.sample_task([t]))

        d_rand_iterator = sampler.get_complete_iterator()

        x_spt, y_spt, x_qry, y_qry = maml.sample_training_data(d_traj_iterators, d_rand_iterator,
                                                               steps=args.update_step, iid=args.iid)
        
        perm = np.random.permutation(args.tasks)
        
        old = []
        for i in range(y_spt.size()[0]):
            num = int(y_spt[i].cpu().numpy())
            if num not in old:
                old.append(num)
            y_spt[i] = torch.tensor(perm[old.index(num)])
            
        for i in range(y_qry.size()[1]):
            num = int(y_qry[0][i].cpu().numpy())
            y_qry[0][i] = torch.tensor(perm[old.index(num)])
        #print('hi', y_qry.size())
        #print('y_spt', y_spt)
        #print('y_qry', y_qry)
        if torch.cuda.is_available():
            x_spt, y_spt, x_qry, y_qry = x_spt.cuda(), y_spt.cuda(), x_qry.cuda(), y_qry.cuda()

        #print('heyyyy', x_spt.size(), y_spt.size(), x_qry.size(), y_qry.size())
        accs, loss = maml(x_spt, y_spt, x_qry, y_qry)

        if step % 1 == 0:
            writer.add_scalar('/metatrain/train/accuracy', accs[-1], step)
            logger.info('step: %d \t training acc %s', step, str(accs))
        if step % 300 == 0:
            correct = 0
            torch.save(maml.net, my_experiment.path + "learner.model")
            for img, target in iterator_test:
                with torch.no_grad():
                    img = img.to(device)
                    target = target.to(device)
                    logits_q = maml.net(img, vars=None, bn_training=False, feature=False)
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct += torch.eq(pred_q, target).sum().item() / len(img)
            writer.add_scalar('/metatrain/test/classifier/accuracy', correct / len(iterator_test), step)
            logger.info("Test Accuracy = %s", str(correct / len(iterator_test)))
            correct = 0
            for img, target in iterator_train:
                with torch.no_grad():
                    
                    img = img.to(device)
                    target = target.to(device)
                    logits_q = maml.net(img, vars=None, bn_training=False, feature=False)
                    pred_q = (logits_q).argmax(dim=1)
                    correct += torch.eq(pred_q, target).sum().item() / len(img)

            logger.info("Train Accuracy = %s", str(correct / len(iterator_train)))
            writer.add_scalar('/metatrain/train/classifier/accuracy', correct / len(iterator_train), step)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--steps', type=int, help='epoch number', default=40000)
    argparser.add_argument('--seed', type=int, help='Seed for random', default=10000)
    argparser.add_argument('--seeds', type=int, nargs='+', help='n way', default=[10])
    argparser.add_argument('--tasks', type=int, help='meta batch size, namely task num', default=1)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-4)
    argparser.add_argument('--meta_feedback_lr', type=float, help='meta-level outer learning rate for feedback weights', default=1e-4)
    argparser.add_argument('--meta_plasticity_lr', type=float, help='meta-level outer learning rate for plasticity', default=1e-4)
    argparser.add_argument('--meta_feedback_strength_lr', type=float, help='meta-level outer learning rate for feedback strength', default=1e-4)

    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=10)
    argparser.add_argument('--name', help='Name of experiment', default="mrcl_classification")
    argparser.add_argument('--dataset', help='Name of experiment', default="omniglot")
    argparser.add_argument("--commit", action="store_true")
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
    argparser.add_argument("--rln", type=int, default=6)
    argparser.add_argument("--rln_end", type=int, default=0)
    argparser.add_argument("--no_class_reset", action="store_true")
    argparser.add_argument("--all_class_reset", action="store_true")
    argparser.add_argument("--from_saved", action="store_true")
    argparser.add_argument("--zero_non_output_plasticity", action="store_true")
    argparser.add_argument("--zero_all_plasticity", action="store_true")
    argparser.add_argument("--model", type=str, default='')
    argparser.add_argument("--width", type=int, default=1024)
    argparser.add_argument("--feedback_l2", type=int, default=0.0)
    argparser.add_argument("--overwrite", action="store_true")
    argparser.add_argument("--optimize_out", action="store_true")
    argparser.add_argument("--plasticity_rank1", action="store_true")
    argparser.add_argument("--freeze_out_plasticity", action="store_true")
    argparser.add_argument("--use_error", action="store_true")
    argparser.add_argument('--imagenet-path', help='Dataset path', default="/data5/jlindsey/continual/miniimagenet")
    argparser.add_argument("--plastic_update", action="store_true")
    argparser.add_argument("--randomize_plastic_weights", action="store_true")
    argparser.add_argument("--zero_plastic_weights", action="store_true")
    argparser.add_argument("--batch_learning", action="store_true")
    argparser.add_argument("--linear_feedback", action="store_true")
    argparser.add_argument("--use_derivative", action="store_true")

    argparser.add_argument("--error_only_to_output", action="store_true")    
    argparser.add_argument("--neuron_level_plasticity", action="store_true")
    argparser.add_argument("--layer_level_plasticity", action="store_true")
    argparser.add_argument("--vary_length", action="store_true")
    argparser.add_argument("--iid", action="store_true")
    argparser.add_argument('--model_type', help='Name of model', default="halfsize")

    argparser.add_argument("--reset_feedback_strength", action="store_true")
    argparser.add_argument("--reset_feedback_vars", action="store_true")


    argparser.add_argument('--inner_plasticity_multiplier', type=float, default=100)






    args = argparser.parse_args()

    args.name = args.name#"/".join([args.dataset, str(args.meta_lr).replace(".", "_"), args.name])
    print(args)
    main(args)
