import argparse
import logging
import random

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.nn import functional as F

import datasets.datasetfactory as df
import model.oja_learner as learner
import model.modelfactory as mf
import utils
from experiment.experiment import experiment
import replay as rep
import datasets.miniimagenet as imgnet



logger = logging.getLogger('experiment')


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    my_experiment = experiment(args.name, args, "./evals/", args.commit)
    writer = SummaryWriter(my_experiment.path + "tensorboard")


    ver = 0

    while os.path.exists(args.modelX + "_" + str(ver)):
        ver += 1
        
    args.modelX = args.modelX + "_" + str(ver-1) + "/learner.model"
    
    logger = logging.getLogger('experiment')
    logger.setLevel(logging.INFO)
    total_clases = 10

    total_ff_vars = 2*(6 + 2 + args.num_extra_dense_layers)

    frozen_layers = []
    for temp in range(args.rln * 2):
        frozen_layers.append("vars." + str(temp))
        

    for temp in range(args.rln_end * 2):
        frozen_layers.append("net.vars." + str(total_ff_vars - 1 - temp))
    #logger.info("Frozen layers = %s", " ".join(frozen_layers))

    #
    final_results_all = []
    
    total_clases = [5]

    if args.twentyclass:
        total_clases = [20]

    if args.twotask:
        total_clases = [2, 10]
    if args.fiftyclass:
        total_clases = [50]
    if args.tenclass:
        total_clases = [10]
    if args.fiveclass:
        total_clases = [5]       
    print('yooo', total_clases)
    for tot_class in total_clases:
        
        avg_perf = 0.0
        print('TOT_CLASS', tot_class)
        lr_list = [0]#[0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]
        for aoo in range(0, args.runs):
            #print('run', aoo)
            keep = np.random.choice(list(range(650)), tot_class, replace=False)
            if args.dataset == "imagenet":
                keep = np.random.choice(list(range(20)), tot_class, replace=False)
                
                dataset = imgnet.MiniImagenet(args.imagenet_path, mode='test', elem_per_class=30, classes=keep, seed=aoo)

                dataset_test = imgnet.MiniImagenet(args.imagenet_path, mode='test', elem_per_class=30, classes=keep, test=args.test, seed=aoo)


                iterator = torch.utils.data.DataLoader(dataset_test, batch_size=128,
                                                       shuffle=True, num_workers=1)
                iterator_sorted = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                       shuffle=False, num_workers=1)
            if args.dataset == "omniglot":

                dataset = utils.remove_classes_omni(
                    df.DatasetFactory.get_dataset("omniglot", train=True, background=False), keep)
                iterator_sorted = torch.utils.data.DataLoader(
                    utils.iterator_sorter_omni(dataset, False, classes=total_clases),
                    batch_size=1,
                    shuffle=False, num_workers=2)
                dataset = utils.remove_classes_omni(
                    df.DatasetFactory.get_dataset("omniglot", train=not args.test, background=False), keep)
                iterator = torch.utils.data.DataLoader(dataset, batch_size=32,
                                                       shuffle=False, num_workers=1)
            elif args.dataset == "CIFAR100":
                keep = np.random.choice(list(range(50, 100)), tot_class)
                dataset = utils.remove_classes(df.DatasetFactory.get_dataset(args.dataset, train=True), keep)
                iterator_sorted = torch.utils.data.DataLoader(
                    utils.iterator_sorter(dataset, False, classes=tot_class),
                    batch_size=16,
                    shuffle=False, num_workers=2)
                dataset = utils.remove_classes(df.DatasetFactory.get_dataset(args.dataset, train=False), keep)
                iterator = torch.utils.data.DataLoader(dataset, batch_size=128,
                                                       shuffle=False, num_workers=1)
            # sampler = ts.MNISTSampler(list(range(0, total_clases)), dataset)
            #
            #print(args)

            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

            results_mem_size = {}

            #print("LEN", len(iterator_sorted))
            for mem_size in [args.memory]:
                max_acc = -10
                max_lr = -10
                for lr in lr_list:
                    #torch.cuda.empty_cache()
                    #print(lr)
                    # for lr in [0.001, 0.0003, 0.0001, 0.00003, 0.00001]:
                    maml = torch.load(args.modelX, map_location='cpu')

                    if args.scratch:
                        config = mf.ModelFactory.get_model(args.model_type, args.dataset)
                        maml = learner.Learner(config, lr)
                        # maml = MetaLearingClassification(args, config).to(device).net

                    #maml.update_lr = lr
                    maml = maml.to(device)

                    for name, param in maml.named_parameters():
                        param.learn = True

                    for name, param in maml.named_parameters():
                        #if name.find("feedback_strength_vars") != -1:
                        #    print(name, param)
                        if name in frozen_layers:
                            # logger.info("Freeezing name %s", str(name))
                            param.learn = False
                            # logger.info(str(param.requires_grad))
                        else:
                            if args.reset:
                                w = nn.Parameter(torch.ones_like(param))
                                # logger.info("W shape = %s", str(len(w.shape)))
                                if len(w.shape) > 1:
                                    torch.nn.init.kaiming_normal_(w)
                                else:
                                    w = nn.Parameter(torch.zeros_like(param))
                                param.data = w
                                param.learn = True

                    frozen_layers = []
                    for temp in range(args.rln * 2):
                        frozen_layers.append("vars." + str(temp))

                    #torch.nn.init.kaiming_normal_(maml.parameters()[-2])
                    #w = nn.Parameter(torch.zeros_like(maml.parameters()[-1]))
                    #maml.parameters()[-1].data = w

                    
                    for n, a in maml.named_parameters():
                        n = n.replace(".", "_")
                        # logger.info("Name = %s", n)
                        if n == "vars_"+str(14+2*args.num_extra_dense_layers):
                            pass
                            #w = nn.Parameter(torch.ones_like(a))
                            # logger.info("W shape = %s", str(w.shape))
                            #torch.nn.init.kaiming_normal_(w)
                            #a.data = w
                        if n == "vars_"+str(15+2*args.num_extra_dense_layers):
                            pass
                            #w = nn.Parameter(torch.zeros_like(a))
                            #a.data = w
                            
                    #for fv in maml.feedback_vars:
                    #    w = nn.Parameter(torch.zeros_like(fv))
                    #    fv.data = w   
                        
                    #for fv in maml.feedback_strength_vars:
                    #    w = nn.Parameter(torch.ones_like(fv))
                    #    fv.data = w                        

                    correct = 0

                    for img, target in iterator:
                        #print('size', target.size())
                        target = torch.tensor(np.array([list(keep).index(int(target.cpu().numpy()[i])) for i in range(target.size()[0])]))
                        with torch.no_grad():
                            img = img.to(device)
                            target = target.to(device)
                            logits_q = maml(img, vars=None, bn_training=False, feature=False)
                            pred_q = (logits_q).argmax(dim=1)
                            correct += torch.eq(pred_q, target).sum().item() / len(img)

                    #logger.info("Pre-epoch accuracy %s", str(correct / len(iterator)))

                    filter_list = ["vars.0", "vars.1", "vars.2", "vars.3", "vars.4", "vars.5"]

                    #logger.info("Filter list = %s", ",".join(filter_list))
                    list_of_names = list(
                        map(lambda x: x[1], list(filter(lambda x: x[0] not in filter_list, maml.named_parameters()))))

                    list_of_params = list(filter(lambda x: x.learn, maml.parameters()))
                    list_of_names = list(filter(lambda x: x[1].learn, maml.named_parameters()))
                    if args.scratch or args.no_freeze:
                        print("Empty filter list")
                        list_of_params = maml.parameters()
                    #
                    #for x in list_of_names:
                    #    logger.info("Unfrozen layer = %s", str(x[0]))
                    opt = torch.optim.Adam(list_of_params, lr=lr)

                    fast_weights = None
                    if args.randomize_plastic_weights:
                        maml.randomize_plastic_weights()
                    if args.zero_plastic_weights:
                        maml.zero_plastic_weights()
                    res_sampler = rep.ReservoirSampler(mem_size)
                    iterator_sorted_new = []
                    iter_count = 0
                    for img, y in iterator_sorted:
                        
                        y = torch.tensor(np.array([list(keep).index(int(y.cpu().numpy()[i])) for i in range(y.size()[0])]))
                        if iter_count % 15 >= args.shots:
                            iter_count += 1
                            continue       
                        iterator_sorted_new.append((img, y))
                        iter_count += 1
                    iterator_sorted = []
                    perm = np.random.permutation(len(iterator_sorted_new))
                    for i in range(len(iterator_sorted_new)):
                        if args.iid:
                            iterator_sorted.append(iterator_sorted_new[perm[i]])
                        else:
                            iterator_sorted.append(iterator_sorted_new[i])

                    for iter in range(0, args.epoch):
                        iter_count = 0
                        imgs = []
                        ys = []
                            
                        for img, y in iterator_sorted:
                            
                            #print('iter count', iter_count)
                            #print('y is', y)
                            


                            #if iter_count % 15 >= args.shots:
                            #    iter_count += 1
                            #    continue
                            iter_count += 1
                            #with torch.no_grad():
                            if args.memory == 0:
                                  img = img.to(device)
                                  y = y.to(device)
                            else:
                                  res_sampler.update_buffer(zip(img, y))
                                  res_sampler.update_observations(len(img))
                                  img = img.to(device)
                                  y = y.to(device)
                                  img2, y2 = res_sampler.sample_buffer(8)
                                  img2 = img2.to(device)
                                  y2 = y2.to(device)
                                  img = torch.cat([img, img2], dim=0)
                                  y = torch.cat([y, y2], dim=0)
                                  #print('img size', img.size())

                            imgs.append(img)
                            ys.append(y)
                            if not args.batch_learning:
                                  logits = maml(img, vars=fast_weights)
                                  fast_weights = maml.getOjaUpdate(y, logits, fast_weights, hebbian=args.hebb)
                        if args.batch_learning:
                              y = torch.cat(ys, 0)
                              img = torch.cat(imgs, 0)
                              logits = maml(img, vars=fast_weights)
                              fast_weights = maml.getOjaUpdate(y, logits, fast_weights, hebbian=args.hebb)


                    #logger.info("Result after one epoch for LR = %f", lr)
                    correct = 0
                    for img, target in iterator:
                        target = torch.tensor(np.array([list(keep).index(int(target.cpu().numpy()[i])) for i in range(target.size()[0])]))
                        img = img.to(device)
                        target = target.to(device)
                        logits_q = maml(img, vars=fast_weights, bn_training=False, feature=False)

                        pred_q = (logits_q).argmax(dim=1)

                        correct += torch.eq(pred_q, target).sum().item() / len(img)

                    #logger.info(str(correct / len(iterator)))
                    if (correct / len(iterator) > max_acc):
                        max_acc = correct / len(iterator)
                        max_lr = lr
                        
                    del maml
                    #del maml
                    #del fast_weights

                lr_list = [max_lr]
                #print('result', max_acc)
                results_mem_size[mem_size] = (max_acc, max_lr)
                #logger.info("Final Max Result = %s", str(max_acc))
                writer.add_scalar('/finetune/best_' + str(aoo), max_acc, tot_class)
                avg_perf += max_acc / args.runs #TODO: change this if/when I ever use memory -- can't choose max memory size differently for each run!
                print('avg perf', avg_perf * args.runs / (1+aoo))
            final_results_all.append((tot_class, results_mem_size))
            #writer.add_scalar('performance', avg_perf, tot_class)
            #print("A=  ", results_mem_size)
            #logger.info("Final results = %s", str(results_mem_size))

            my_experiment.results["Final Results"] = final_results_all
            my_experiment.store_json()
            np.save('evals/final_results_'+args.orig_name+'.npy', final_results_all) 
            #print("FINAL RESULTS = ", final_results_all)
    writer.close()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=1)
    argparser.add_argument('--seed', type=int, help='epoch number', default=222)
    argparser.add_argument('--memory', type=int, help='epoch number', default=0)
    argparser.add_argument('--modelX', type=str, help='epoch number', default="none")
    argparser.add_argument('--scratch', action='store_true', default=False)
    argparser.add_argument('--dataset', help='Name of experiment', default="omniglot")
    argparser.add_argument('--name', help='Name of experiment', default="evaluation")
    argparser.add_argument("--commit", action="store_true")
    argparser.add_argument("--no-freeze", action="store_true")
    argparser.add_argument('--reset', action="store_true")
    argparser.add_argument('--test', action="store_true")
    argparser.add_argument("--iid", action="store_true")
    argparser.add_argument("--rln", type=int, default=6)
    argparser.add_argument("--rln_end", type=int, default=0)
    argparser.add_argument("--shots", type=int, default=1000000)
    argparser.add_argument("--runs", type=int, default=50)
    argparser.add_argument("--from_saved", action="store_true")
    argparser.add_argument("--model", type=str, default='')
    argparser.add_argument("--overwrite", action="store_true")
    argparser.add_argument("--fiveclass", action="store_true")
    argparser.add_argument("--tenclass", action="store_true", default=True)


    argparser.add_argument("--twentyclass", action="store_true", default=True)
    argparser.add_argument("--fiftyclass", action="store_true")


    argparser.add_argument('--num_extra_dense_layers', type=int, help='num dense layers in addition to one intermediate layer', default=0)
    argparser.add_argument("--use_error", action="store_true")
    argparser.add_argument("--twotask", action="store_true")
    argparser.add_argument("--randomize_plastic_weights", action="store_true")
    argparser.add_argument("--zero_plastic_weights", action="store_true")
    argparser.add_argument("--batch_learning", action="store_true")
    argparser.add_argument("--linear_feedback", action="store_true")
    argparser.add_argument("--use_derivative", action="store_true")
    argparser.add_argument('--hebb', action="store_true")
    argparser.add_argument('--num_feedback_layers', type=int, default=1)
    argparser.add_argument("--neuron_level_plasticity", action="store_true")


    argparser.add_argument("--layer_level_plasticity", action="store_true")


    argparser.add_argument('--imagenet-path', help='Dataset path', default="/home/jwl2182/continual/miniimagenet")


    argparser.add_argument('--model_type', help='Name of model', default="na")





    args = argparser.parse_args()

    import os

    args.orig_name = args.name
    args.name = "/".join([args.dataset, "eval", str(args.epoch).replace(".", "_"), args.name])

    main(args)
