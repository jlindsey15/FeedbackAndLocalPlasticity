import logging

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import model.learner as Learner

logger = logging.getLogger("experiment")


class MetaLearingClassification(nn.Module):
    """
    MetaLearingClassification Learner
    """

    def __init__(self, args, config):
        print('initializing MLearner!')

        super(MetaLearingClassification, self).__init__()
        
        self.init_stuff(args)

        self.net = Learner.Learner(config)

        
        #print(self.net.parameters())
        #print('hey')
        #print(self.net.vars)
        #sys.exit()
        self.init_opt()
        
    def init_stuff(self, args):

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.update_step = args.update_step
        self.train_on_new = args.train_on_new
        self.plastic_update = args.plastic_update
        
    def init_opt(self):
        self.optimizer = optim.Adam(self.net.vars, lr=self.meta_lr)
    

    def reset_classifer(self, class_to_reset):
        bias = self.net.parameters()[-1]
        weight = self.net.parameters()[-2]
        torch.nn.init.kaiming_normal_(weight[class_to_reset].unsqueeze(0))

    def reset_layer(self):
        bias = self.net.parameters()[-1]
        weight = self.net.parameters()[-2]
        torch.nn.init.kaiming_normal_(weight)

    def sample_training_data(self, iterators, it2, steps=2):

        x_traj = []
        y_traj = []
        x_rand = []
        y_rand = []

        counter = 0

        class_cur = 0
        class_to_reset = 0
        for it1 in iterators:
            for img, data in it1:
                # y_mapping[class_cur] = float(y_mapping[class_cur])
                class_to_reset = data[0].item()
                # data[data>-1] = y_mapping[class_cur]
                counter += 1
                x_traj.append(img)
                y_traj.append(data)
                if counter % int(steps / len(iterators)) == 0:
                    class_cur += 1
                    break
        self.reset_classifer(class_to_reset)

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

        y_rand_temp = torch.cat(y_rand_temp).unsqueeze(0)
        x_rand_temp = torch.cat(x_rand_temp).unsqueeze(0)
        x_traj, y_traj, x_rand, y_rand = torch.stack(x_traj), torch.stack(y_traj), torch.stack(x_rand), torch.stack(
            y_rand)

        x_rand = torch.cat([x_rand, x_rand_temp], 1)
        y_rand = torch.cat([y_rand, y_rand_temp], 1)

        if self.train_on_new:
            return x_traj, y_traj, x_rand_temp, y_rand_temp #equivalent to x_traj, y_traj, x_traj, y_traj
        
        else:
            return x_traj, y_traj, x_rand, y_rand #x_rand has both the randomly sampled points and the traj points, in that order



    def forward(self, x_traj, y_traj, x_rand, y_rand):
        """

        :param x_traj:   [b, setsz, c_, h, w]
        :param y_traj:   [b, setsz]
        :param x_rand:   [b, querysz, c_, h, w]
        :param y_rand:   [b, querysz]
        :return:
        """
        print('heyyyy', x_traj.size(), y_traj.size(), x_rand.size(), y_rand.size())


        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]

        for i in range(1):
            logits = self.net(x_traj[0], vars=None, bn_training=False)
            loss = F.cross_entropy(logits, y_traj[0])
            grad = torch.autograd.grad(loss, self.net.parameters())
            # fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            if self.plastic_update:
                fast_weights = list(
                    map(lambda p: p[1] - self.update_lr * p[0] * p[2] if p[1].learn else p[1], zip(grad, self.net.vars, self.net.vars_plasticity)))       
            else:
                fast_weights = list(
                    map(lambda p: p[1] - self.update_lr * p[0] if p[1].learn else p[1], zip(grad, self.net.parameters())))
            for params_old, params_new in zip(self.net.parameters(), fast_weights):
                params_new.learn = params_old.learn

            # this is the loss and accuracy before first update
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
                grad = torch.autograd.grad(loss, fast_weights)

                if self.plastic_update:
                    fast_weights = list(
                        map(lambda p: p[1] - self.update_lr * p[0] * p[2] if p[1].learn else p[1], zip(grad, fast_weights, self.net.vars_plasticity)))       
                else:
                    fast_weights = list(
                        map(lambda p: p[1] - self.update_lr * p[0] if p[1].learn else p[1], zip(grad, fast_weights)))

                for params_old, params_new in zip(self.net.parameters(), fast_weights):
                    params_new.learn = params_old.learn

                logits = self.net(x_rand[0], fast_weights, bn_training=False)
                loss_q = F.cross_entropy(logits, y_rand[0])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_rand[0]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

        self.optimizer.zero_grad()
        loss_q = losses_q[-1]
        loss_q.backward()

        self.optimizer.step()
        accs = np.array(corrects) / len(x_rand[0])

        return accs, loss


class MetaLearnerRegression(nn.Module):
    """
    MetaLearingClassification Learner
    """
    
    def __init__(self, args, config):

        super(MetaLearnerRegression, self).__init__()

        
        self.init_stuff(args)

        self.net = Learner.Learner(config)

        
        #print(self.net.parameters())
        #print('hey')
        #print(self.net.vars)
        #sys.exit()
        self.init_opt()
        
    def init_stuff(self, args):

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.update_step = args.update_step
        self.train_on_new = args.train_on_new
        self.plastic_update = args.plastic_update
        
    def init_opt(self):
        self.optimizer = optim.Adam(self.net.vars, lr=self.meta_lr)
        self.meta_optim = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [1500, 2500, 3500], 0.3)



    def forward(self, x_traj, y_traj, x_rand, y_rand):

        losses_q = [0 for _ in range(len(x_traj) + 1)]

        for i in range(1):
            logits = self.net(x_traj[0], vars=None, bn_training=False)
            logits_select = []
            for no, val in enumerate(y_traj[0, :, 1].long()):
                logits_select.append(logits[no, val])
            logits = torch.stack(logits_select).unsqueeze(1)
            loss = F.mse_loss(logits, y_traj[0, :, 0].unsqueeze(1))
            grad = torch.autograd.grad(loss, self.net.parameters())

            if self.plastic_update:
                fast_weights = list(
                    map(lambda p: p[1] - self.update_lr * p[0] * p[2] if p[1].learn else p[1], zip(grad, self.net.vars, self.net.vars_plasticity)))       
            else:
                fast_weights = list(
                    map(lambda p: p[1] - self.update_lr * p[0] if p[1].learn else p[1], zip(grad, self.net.parameters())))
            for params_old, params_new in zip(self.net.parameters(), fast_weights):
                params_new.learn = params_old.learn

            with torch.no_grad():

                logits = self.net(x_rand[0], vars=None, bn_training=False)

                logits_select = []
                for no, val in enumerate(y_rand[0, :, 1].long()):
                    logits_select.append(logits[no, val])
                logits = torch.stack(logits_select).unsqueeze(1)
                loss_q = F.mse_loss(logits, y_rand[0, :, 0].unsqueeze(1))
                losses_q[0] += loss_q

            for k in range(1, len(x_traj)):
                logits = self.net(x_traj[k], fast_weights, bn_training=False)

                logits_select = []
                for no, val in enumerate(y_traj[k, :, 1].long()):
                    logits_select.append(logits[no, val])
                logits = torch.stack(logits_select).unsqueeze(1)

                loss = F.mse_loss(logits, y_traj[k, :, 0].unsqueeze(1))
                grad = torch.autograd.grad(loss, fast_weights)
                if self.plastic_update:
                    fast_weights = list(
                        map(lambda p: p[1] - self.update_lr * p[0] * p[2] if p[1].learn else p[1], zip(grad, fast_weights, self.net.vars_plasticity)))       
                else:
                    fast_weights = list(
                        map(lambda p: p[1] - self.update_lr * p[0] if p[1].learn else p[1], zip(grad, fast_weights)))

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

        loss_q = losses_q[k + 1]
        loss_q.backward()

        self.optimizer.step()

        return losses_q


def main():
    pass


if __name__ == '__main__':
    main()
