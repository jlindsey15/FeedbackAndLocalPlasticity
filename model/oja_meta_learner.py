import logging

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.nn import DataParallel
import sys

import model.oja_learner as Learner #note it's the oja learner!!!

import model.oja_learner_regression as LearnerRegression #note it's the oja learner!!!



logger = logging.getLogger("experiment")


class MetaLearingClassification(nn.Module):
    """
    MetaLearingClassification Learner
    """

    def __init__(self, args, config):

        super(MetaLearingClassification, self).__init__()
        
        self.init_stuff(args)

        self.net = Learner.Learner(config, args.num_feedback_layers, args.init_plasticity, args.feedback_strength, width=args.width, feedback_l2=args.feedback_l2, optimize_out=args.optimize_out, use_error=args.use_error, linear_feedback=args.linear_feedback, use_derivative=args.use_derivative, error_only_to_output=args.error_only_to_output, neuron_level_plasticity=args.neuron_level_plasticity, layer_level_plasticity=args.layer_level_plasticity, inner_plasticity_multiplier=args.inner_plasticity_multiplier)
        
        #print(self.net.parameters())
        #print('hey')
        #print(self.net.vars)
        #sys.exit()
        self.init_opt()
        
    def init_stuff(self, args):

        self.hebb = args.hebb #use plain Hebbian learning instead of Oja's rule
        
        #self.feedback_strength_clamp = args.feedback_strength_clamp
        self.train_on_new = args.train_on_new
        #self.propagate_feedback = args.propagate_feedback
        
        self.num_feedback_layers = args.num_feedback_layers
        
        #self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.meta_feedback_lr = args.meta_feedback_lr
        self.meta_plasticity_lr = args.meta_plasticity_lr
        self.meta_feedback_strength_lr = args.meta_feedback_strength_lr
        
        self.no_class_reset = args.no_class_reset
        self.all_class_reset = args.all_class_reset

        #self.optimize_feedback = args.optimize_feedback
        self.update_step = args.update_step
        self.use_error = args.use_error
        self.randomize_plastic_weights = args.randomize_plastic_weights
        self.zero_plastic_weights = args.zero_plastic_weights
        self.batch_learning = args.batch_learning
        self.linear_feedback = args.linear_feedback
        self.use_derivative = args.use_derivative
        
        self.vary_length = args.vary_length
        
        self.neuron_level_plasticity = args.neuron_level_plasticity
        
        self.layer_level_plasticity = args.layer_level_plasticity


        
        if self.batch_learning:
            self.update_step = 1
        
    def init_opt(self):
        self.optimizer = optim.Adam(self.net.vars, lr=self.meta_lr)
    
        self.feedback_optimizer = optim.Adam(self.net.feedback_vars, lr=self.meta_feedback_lr)
        
        if self.neuron_level_plasticity:
            self.plasticity_optimizer = optim.Adam(self.net.neuron_plasticity, lr=self.meta_plasticity_lr)
            
        elif self.layer_level_plasticity:
            self.plasticity_optimizer = optim.Adam(self.net.layer_plasticity, lr=self.meta_plasticity_lr)
        else:
            self.plasticity_optimizer = optim.Adam(self.net.plasticity, lr=self.meta_plasticity_lr)
        
        self.feedback_strength_optimizer = optim.Adam(self.net.feedback_strength_vars, lr=self.meta_feedback_strength_lr)

    def reset_classifer(self, class_to_reset):
        bias = self.net.parameters()[-1]
        weight = self.net.parameters()[-2]
        torch.nn.init.kaiming_normal_(weight[class_to_reset].unsqueeze(0))

    def reset_layer(self, layer=0):
        bias = self.net.parameters()[-1-(2*layer)]
        weight = self.net.parameters()[-2-(2*layer)]
        torch.nn.init.kaiming_normal_(weight)

    def sample_training_data(self, iterators, it2, steps=2, iid=False):

        x_traj = []
        y_traj = []
        x_rand = []
        y_rand = []

        counter = 0

        class_cur = 0
        class_to_reset = 0
        for it1 in iterators:
            for img, data in it1:
                #print('sampling img size', img.size(), data.size())


                # y_mapping[class_cur] = float(y_mapping[class_cur])
                class_to_reset = data[0].item()
                #if self.all_class_reset:
                #    self.reset_layer()
                #elif not self.no_class_reset:
                    #print('resetting classifier', class_to_reset)
                #    self.reset_classifer(class_to_reset)
                # data[data>-1] = y_mapping[class_cur]
                counter += 1
                x_traj.append(img)
                y_traj.append(data)
                if counter % int(steps / len(iterators)) == 0:
                    class_cur += 1
                    break
        #self.reset_layer()          
        #if self.all_class_reset:
        #    self.reset_layer()
        #elif not self.no_class_reset:
        #    print('resetting classifier')
        #    self.reset_classifer(class_to_reset)

        if len(x_traj) < steps:
            it1 = iterators[-1]
            for img, data in it1:
                counter += 1

                x_traj.append(img)
                y_traj.append(data)
                print("Len of iterators = ", len(iterators))
                if counter % int(steps % len(iterators)) == 0:
                    break

        counter = 0
        for img, data in it2:
            if counter == 1:
                break
            x_rand.append(img)
            y_rand.append(data)
            counter += 1

        class_cur = 0
        counter = 0
        x_rand_temp = []
        y_rand_temp = []
        for it1 in iterators:
            for img, data in it1:
                counter += 1
                x_rand_temp.append(img)
                y_rand_temp.append(data)
                if counter % int(steps / len(iterators)) == 0:
                    class_cur += 1
                    break

        rand_indices = list(range(len(x_traj)))
        np.random.shuffle(rand_indices)
        if iid:
            x_traj_new = []
            y_traj_new = []
            for a in rand_indices:
                x_traj_new.append(x_traj[a])
                y_traj_new.append(y_traj[a])
            x_traj = x_traj_new
            y_traj = y_traj_new
        #print('Sizes', x_rand_temp.size(), y_rand_temp.size(), x_traj.size(), y_traj.size(), x_rand.size(), y_rand.size())
        y_rand_temp = torch.cat(y_rand_temp).unsqueeze(0)
        x_rand_temp = torch.cat(x_rand_temp).unsqueeze(0)
        x_traj, y_traj, x_rand, y_rand = torch.stack(x_traj), torch.stack(y_traj), torch.stack(x_rand), torch.stack(
            y_rand)

        #print('new Sizes', x_rand_temp.size(), y_rand_temp.size(), x_traj.size(), y_traj.size(), x_rand.size(), y_rand.size())
        
        
        x_rand = torch.cat([x_rand, x_rand_temp], 1)
        y_rand = torch.cat([y_rand, y_rand_temp], 1)

        #print('new new Sizes', x_rand_temp.size(), y_rand_temp.size(), x_traj.size(), y_traj.size(), x_rand.size(), y_rand.size())
        
        if self.train_on_new:
            #print('diffx', torch.sum(torch.abs(x_traj.view(-1) - x_rand_temp.view(-1))))
            #print('diffy', torch.sum(torch.abs(y_traj.view(-1) - y_rand_temp.view(-1))))
            #print('y_traj', y_traj)
            #print('y_rand_temp', y_rand_temp)
            return x_traj, y_traj, x_rand_temp, y_rand_temp #equivalent to x_traj, y_traj, x_traj, y_traj
        
        else:
            return x_traj, y_traj, x_rand, y_rand #x_rand has both the randomly sampled points and the traj points, in that order

    def forward(self, x_traj, y_traj, x_rand, y_rand):
        #print('heyy', x_rand.size())
        """

        :param x_traj:   [b, setsz, c_, h, w]
        :param y_traj:   [b, setsz]
        :param x_rand:   [b, querysz, c_, h, w]
        :param y_rand:   [b, querysz]
        :return:
        """
        
        #if self.batch_learning:
        #    x_traj = torch.transpose(x_traj, 0, 1)
        #    y_traj = torch.transpose(y_traj, 0, 1)
        
        #print('heyy', x_traj.size(), y_traj.size(), x_rand.size(), y_rand.size())

        if self.vary_length:
            self.update_step = len(x_traj)
        
        if self.randomize_plastic_weights:
            self.net.randomize_plastic_weights()
        if self.zero_plastic_weights:
            self.net.zero_plastic_weights()

        
        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]

        for i in range(1):
            if self.batch_learning:
                logits = self.net(x_traj[:, 0], vars=None, bn_training=False)
                loss = F.cross_entropy(logits, y_traj[:, 0])
                #grad = torch.autograd.grad(loss, self.net.parameters())

                # fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

                #fast_weights = list(
                #    map(lambda p: p[1] - self.update_lr * p[0] if p[1].learn else p[1], zip(grad, self.net.parameters())))

                fast_weights = self.net.getOjaUpdate(y_traj[:, 0], logits, None, hebbian=self.hebb)
            else:
                logits = self.net(x_traj[0], vars=None, bn_training=False)
                loss = F.cross_entropy(logits, y_traj[0])
                #grad = torch.autograd.grad(loss, self.net.parameters())

                # fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

                #fast_weights = list(
                #    map(lambda p: p[1] - self.update_lr * p[0] if p[1].learn else p[1], zip(grad, self.net.parameters())))

                fast_weights = self.net.getOjaUpdate(y_traj[0], logits, None, hebbian=self.hebb)
            for params_old, params_new in zip(self.net.parameters(), fast_weights):
                #print('yooo', params_old, params_old.learn)
                params_new.learn = params_old.learn

            # this is the loss and accuracy before first update
            
            if self.batch_learning:
                logits_q = self.net(x_rand[0], self.net.parameters(), bn_training=False)
                loss_q = F.cross_entropy(logits_q, y_rand[0])
                losses_q[0] += loss_q
                with torch.no_grad():


                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_rand[0]).sum().item()
                    corrects[0] = corrects[0] + correct
                logits_q = self.net(x_rand[0], fast_weights, bn_training=False)
                loss_q = F.cross_entropy(logits_q, y_rand[0])
                losses_q[1] += loss_q
                with torch.no_grad():
                    # [setsz, nway]


                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_rand[0]).sum().item()
                    corrects[1] = corrects[1] + correct
                    
            else:
                with torch.no_grad():
                    logits_q = self.net(x_rand[0], self.net.parameters(), bn_training=False)
                    loss_q = F.cross_entropy(logits_q, y_rand[0])
                    losses_q[0] += loss_q

                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_rand[0]).sum().item()
                    corrects[0] = corrects[0] + correct

                with torch.no_grad():
                    # [setsz, nway]
                    logits_q = self.net(x_rand[0], fast_weights, bn_training=False)
                    loss_q = F.cross_entropy(logits_q, y_rand[0])
                    losses_q[1] += loss_q

                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_rand[0]).sum().item()
                    corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):

                logits = self.net(x_traj[k], fast_weights, bn_training=False)
                loss = F.cross_entropy(logits, y_traj[k])
                #grad = torch.autograd.grad(loss, fast_weights)
                

                #fast_weights = list(
                #    map(lambda p: p[1] - self.update_lr * p[0] if p[1].learn else p[1], zip(grad, fast_weights)))
                
                fast_weights = self.net.getOjaUpdate(y_traj[k], logits, fast_weights)

                for params_old, params_new in zip(self.net.parameters(), fast_weights):
                    params_new.learn = params_old.learn

                logits = self.net(x_rand[0], fast_weights, bn_training=False)
                
                loss_q = F.cross_entropy(logits, y_rand[0])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits, dim=1).argmax(dim=1)
                    #print('howdy', pred_q, y_rand[0])
                    correct = torch.eq(pred_q, y_rand[0]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

        
        self.optimizer.zero_grad()
        self.feedback_optimizer.zero_grad()
        self.plasticity_optimizer.zero_grad()
        self.feedback_strength_optimizer.zero_grad()
        
        loss_q = losses_q[-1]
        loss_q.backward()
        '''
        print("PLASTICITY")
        for p in self.net.plasticity:
          try:
            print(p.shape, torch.sum(torch.abs(p)), torch.sum(torch.abs(p.grad)))
          except:
            print("None")
        '''
        '''
        print("GRADIENTS")
        for p in self.net.feedback_vars:
          try:
            print(p.shape, torch.sum(torch.abs(p)), torch.sum(torch.abs(p.grad)))
          except:
            print("None")
        print("PLASTICITY")
        for p in self.net.plasticity:
          try:
            print(p.shape, torch.sum(torch.abs(p)), torch.sum(torch.abs(p.grad)))
          except:
            print("None")
        print("REGULAR")
        for p in self.net.vars:
          try:
            print(p.shape, torch.sum(torch.abs(p)), torch.sum(torch.abs(p.grad)))
          except:
            print("None")
        #sys.exit()
        '''
        
        
        #self.optimizer.zero_grad()
        #self.feedback_optimizer.zero_grad()
        #self.plasticity_optimizer.zero_grad()
        #self.feedback_strength_optimizer.zero_grad()
        
        for p in self.net.vars:
            if p.grad is not None:
                p.grad[torch.isnan(p.grad)] = 0
                #print('yo1', torch.sum(torch.isnan(p.grad)))
        for p in self.net.feedback_vars:
            #print('feedback', torch.sum(torch.abs(p)))
            if p.grad is not None:
                p.grad[torch.isnan(p.grad)] = 0
        for p in self.net.feedback_strength_vars:
            #print('feedback strength', p)
            if p.grad is not None:
                p.grad[torch.isnan(p.grad)] = 0  
        for p in self.net.plasticity:
            if p.grad is not None:
                p.grad[torch.isnan(p.grad)] = 0   
        for p in self.net.layer_plasticity:
            if p.grad is not None:
                p.grad[torch.isnan(p.grad)] = 0 
        for p in self.net.neuron_plasticity:
            if p.grad is not None:
                p.grad[torch.isnan(p.grad)] = 0 
                
        self.optimizer.step()
        
        self.feedback_optimizer.step()
        
        self.plasticity_optimizer.step()
        
        self.feedback_strength_optimizer.step()
        
        '''
        print("PLASTICITY AFTER")
        for p in self.net.plasticity:
          try:
            print(p.shape, torch.sum(torch.abs(p)), torch.sum(torch.abs(p.grad)))
          except:
            print("None")
        '''
        accs = np.array(corrects) / len(x_rand[0])

        return accs, loss


class MetaLearnerRegression(nn.Module):
    """
    MetaLearingClassification Learner
    """
    
    def __init__(self, args, config):

        super(MetaLearnerRegression, self).__init__()

        
        self.init_stuff(args)

        self.net = LearnerRegression.Learner(config, args.num_feedback_layers, args.init_plasticity, args.feedback_strength, width=args.width, feedback_l2=args.feedback_l2, optimize_out=args.optimize_out, use_error=args.use_error, feedback_only_to_output=args.feedback_only_to_output, error_only_to_output=args.error_only_to_output, linear_feedback=args.linear_feedback, use_derivative=args.use_derivative, neuron_level_plasticity=args.neuron_level_plasticity, layer_level_plasticity=args.layer_level_plasticity, coarse_level_plasticity=args.coarse_level_plasticity)
        
        #print(self.net.parameters())
        #print('hey')
        #print(self.net.vars)
        #sys.exit()
        self.init_opt()

    def init_stuff(self, args):

        self.hebb = args.hebb #use plain Hebbian learning instead of Oja's rule
        
        #self.feedback_strength_clamp = args.feedback_strength_clamp
        self.train_on_new = args.train_on_new
        #self.propagate_feedback = args.propagate_feedback
        
        self.num_feedback_layers = args.num_feedback_layers
        
        #self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.meta_feedback_lr = args.meta_feedback_lr
        self.meta_plasticity_lr = args.meta_plasticity_lr
        self.meta_feedback_strength_lr = args.meta_feedback_strength_lr
        
        self.no_class_reset = args.no_class_reset
        self.all_class_reset = args.all_class_reset

        #self.optimize_feedback = args.optimize_feedback
        self.update_step = args.update_step
        self.use_error = args.use_error
        self.randomize_plastic_weights = args.randomize_plastic_weights
        self.zero_plastic_weights = args.zero_plastic_weights
        
        self.linear_feedback = args.linear_feedback
        self.use_derivative = args.use_derivative
        
        self.neuron_level_plasticity = args.neuron_level_plasticity
        
        self.layer_level_plasticity = args.layer_level_plasticity

        self.coarse_level_plasticity = args.layer_level_plasticity


        
        self.vary_length = args.vary_length

        
    def init_opt(self):
        self.optimizer = optim.Adam(self.net.vars, lr=self.meta_lr)
    
        self.feedback_optimizer = optim.Adam(self.net.feedback_vars, lr=self.meta_feedback_lr)
        
        if self.neuron_level_plasticity:
            self.plasticity_optimizer = optim.Adam(self.net.neuron_plasticity, lr=self.meta_plasticity_lr)
        elif self.layer_level_plasticity:
            self.plasticity_optimizer = optim.Adam(self.net.layer_plasticity, lr=self.meta_plasticity_lr)
        elif self.coarse_level_plasticity:
            self.plasticity_optimizer = optim.Adam([self.net.coarse_plasticity], lr=self.meta_plasticity_lr)
        else:
            self.plasticity_optimizer = optim.Adam(self.net.plasticity, lr=self.meta_plasticity_lr)        
        self.feedback_strength_optimizer = optim.Adam(self.net.feedback_strength_vars, lr=self.meta_feedback_strength_lr)
        self.meta_optim = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [1500, 2500, 3500], 0.3)




    def forward(self, x_traj, y_traj, x_rand, y_rand):
        #print('sizes', x_traj.size(), y_traj.size(), x_rand.size(), y_rand.size(), y_traj)
        #print('traj length', len(x_traj), x_traj[0].size())
        
        
        #if self.vary_length:
        #    self.update_step = len(x_traj)

            
        if self.randomize_plastic_weights:
            self.net.randomize_plastic_weights()
        if self.zero_plastic_weights:
            self.net.zero_plastic_weights()
            
        
        losses_q = [0 for _ in range(len(x_traj) + 1)]

        for i in range(1):
            logits = self.net(x_traj[0], vars=None, bn_training=False)
            #print('logits before', logits.size())
            logits_select = []
            for no, val in enumerate(y_traj[0, :, 1].long()):
                logits_select.append(logits[no, val])
            
            logits = torch.stack(logits_select).unsqueeze(1)
            #print('logits after', logits.size())
            loss = F.mse_loss(logits, y_traj[0, :, 0].unsqueeze(1))
            fast_weights = self.net.getOjaUpdate(y_traj[0, :, 0:1], logits, None, hebbian=self.hebb)
            for params_old, params_new in zip(self.net.parameters(), fast_weights):
                #print('yooo', params_old.name, params_old.learn)
                params_new.learn = params_old.learn
                
            with torch.no_grad():

                logits = self.net(x_rand[0], self.net.parameters(), bn_training=False)

                logits_select = []
                for no, val in enumerate(y_rand[0, :, 1].long()):
                    logits_select.append(logits[no, val])
                logits = torch.stack(logits_select).unsqueeze(1)
                loss_q = F.mse_loss(logits, y_rand[0, :, 0].unsqueeze(1))
                losses_q[0] += loss_q
                

            logits = self.net(x_rand[0], fast_weights, bn_training=False)

            logits_select = []
            for no, val in enumerate(y_rand[0, :, 1].long()):
                logits_select.append(logits[no, val])
            logits = torch.stack(logits_select).unsqueeze(1)
            loss_q = F.mse_loss(logits, y_rand[0, :, 0].unsqueeze(1))
            losses_q[1] += loss_q
            k = 0
            for k in range(1, len(x_traj)):
                logits = self.net(x_traj[k], fast_weights, bn_training=False)

                logits_select = []
                for no, val in enumerate(y_traj[k, :, 1].long()):
                    logits_select.append(logits[no, val])
                logits = torch.stack(logits_select).unsqueeze(1)

                loss = F.mse_loss(logits, y_traj[k, :, 0].unsqueeze(1))
                fast_weights = self.net.getOjaUpdate(y_traj[k, :, 0:1], logits, fast_weights)

                for params_old, params_new in zip(self.net.parameters(), fast_weights):
                    params_new.learn = params_old.learn

                logits_q = self.net(x_rand[0, 0:int((k + 1) * len(x_rand[0]) / len(x_traj)), :], fast_weights,
                                    bn_training=False)

                logits_select = []
                for no, val in enumerate(y_rand[0, 0:int((k + 1) * len(x_rand[0]) / len(x_traj)), 1].long()):
                    logits_select.append(logits_q[no, val])
                logits = torch.stack(logits_select).unsqueeze(1)
                loss_q = F.mse_loss(logits, y_rand[0, 0:int((k + 1) * len(x_rand[0]) / len(x_traj)), 0].unsqueeze(1))

                losses_q[k + 1] += loss_q

        self.optimizer.zero_grad()
        self.feedback_optimizer.zero_grad()
        self.plasticity_optimizer.zero_grad()
        self.feedback_strength_optimizer.zero_grad()
        
        loss_q = losses_q[k + 1]

        loss_q.backward()
        
        '''
        for p in self.net.vars:
            if p.grad is not None:
                p.grad[torch.isnan(p.grad)] = 0
                #print('yo1', torch.sum(torch.isnan(p.grad)))
        for p in self.net.feedback_vars:
            if p.grad is not None:
                p.grad[torch.isnan(p.grad)] = 0
        for p in self.net.feedback_strength_vars:
            if p.grad is not None:
                p.grad[torch.isnan(p.grad)] = 0  
        for p in self.net.plasticity:
            if p.grad is not None:
                p.grad[torch.isnan(p.grad)] = 0   
        for p in self.net.layer_plasticity:
            if p.grad is not None:
                p.grad[torch.isnan(p.grad)] = 0 
        for p in self.net.neuron_plasticity:
            if p.grad is not None:
                p.grad[torch.isnan(p.grad)] = 0 
                
        '''        
        self.optimizer.step()
        
        self.feedback_optimizer.step()
        
        self.plasticity_optimizer.step()
        
        self.feedback_strength_optimizer.step()

        return losses_q


def main():
    pass


if __name__ == '__main__':
    main()
