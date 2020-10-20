import logging

import torch
from torch import nn
from torch.nn import functional as F
import sys

logger = logging.getLogger("experiment")


class Learner(nn.Module):
    """

    """
    
    def randomize_plastic_weights(self):

        idx = 0
        layer = 0
        for i, (name, param) in enumerate(self.config):
            if name == 'conv2d':
              idx += 2
              layer += 1
            elif name == 'convt2d':
              idx += 2
              layer += 1
            elif name == 'linear':
              if self.vars[idx].learn:
                #print('randomizing: ', idx)
                torch.nn.init.kaiming_normal_(self.vars[idx])
                torch.nn.init.zeros_(self.vars[idx+1])
                self.vars[idx].learn = True
              idx += 2
              layer += 1
            elif name == 'bn':
              idx += 2
                    
    def zero_plastic_weights(self):
        
        idx = 0
        layer = 0
        for i, (name, param) in enumerate(self.config):
            if name == 'conv2d':
              idx += 2
              layer += 1
            elif name == 'convt2d':
              idx += 2
              layer += 1
            elif name == 'linear':
              #w, b = self.vars[idx], self.vars[idx + 1]
              if self.vars[idx].learn:
                #print('zeroing: ', idx)
                torch.nn.init.zeros_(self.vars[idx])
                torch.nn.init.zeros_(self.vars[idx+1])
                self.vars[idx].learn = True
              idx += 2
              layer += 1
            elif name == 'bn':
              idx += 2


    def __init__(self, config, num_feedback_layers, init_plasticity, init_feedback_strength, width=1024, feedback_l2=0.0, optimize_out=False, use_error=False, linear_feedback=False, use_derivative=False, error_only_to_output=False, neuron_level_plasticity=False, layer_level_plasticity=False, inner_plasticity_multiplier=1):
        """

        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__()

        self.config = config
        self.width = width

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        self.vars_plasticity = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()
        self.feedback_strength_vars = nn.ParameterList()

        self.activations_list = []
        self.activations_list.append([])
        self.feedback_strength_vars.append(nn.Parameter(torch.zeros(1)))
        self.init_plasticity = init_plasticity
        self.init_feedback_strength = init_feedback_strength
        
        self.num_feedback_layers = num_feedback_layers
        self.feedback_vars = nn.ParameterList()
        self.feedback_vars_bundled = []
        self.optimize_out = optimize_out
        self.use_error = use_error
        self.linear_feedback = linear_feedback
        self.use_derivative = use_derivative
        self.error_only_to_output = error_only_to_output   

        self.neuron_level_plasticity = neuron_level_plasticity
        
        self.layer_level_plasticity = layer_level_plasticity
        
        self.inner_plasticity_multiplier = inner_plasticity_multiplier
        
        num_outputs = self.config[-1][1][0]

        self.plasticity = nn.ParameterList()
        
        
        self.neuron_plasticity = nn.ParameterList()
 
        self.layer_plasticity = nn.ParameterList()


        starting_width = 84
        cur_width = starting_width
        for i, (name, param) in enumerate(self.config):
            if name is 'conv2d':
                stride=param[4]
                padding=param[5]
                
                #print('cur_width', cur_width, param[3])
                cur_width = (cur_width + 2*padding - param[3] + stride) // stride
                #print('new cur_width', cur_width)
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                
                self.vars_plasticity.append(nn.Parameter(torch.ones(*param[:4])))
                self.vars_plasticity.append(nn.Parameter(torch.ones(param[0])))
                #self.activations_list.append([])
                self.plasticity.append(nn.Parameter(self.init_plasticity * torch.ones(param[0], param[1]*param[2]*param[3]))) #not implemented
                self.neuron_plasticity.append(nn.Parameter(torch.zeros(1))) #not implemented

                self.layer_plasticity.append(nn.Parameter(self.init_plasticity * torch.ones(1))) #not implemented

                feedback_var = []
                #'''
                
                for fl in range(num_feedback_layers):
                    in_dim = self.width
                    out_dim = self.width
                    if fl == num_feedback_layers - 1:
                        out_dim = param[0] * cur_width * cur_width
                    if fl == 0:
                        in_dim = num_outputs
                    feedback_w_shape = [out_dim, in_dim]
                    feedback_w = nn.Parameter(torch.ones(feedback_w_shape))
                    feedback_b =  nn.Parameter(torch.zeros(out_dim))
                    torch.nn.init.kaiming_normal_(feedback_w)
                    feedback_var.append((feedback_w, feedback_b))
                    self.feedback_vars.append(feedback_w)
                    self.feedback_vars.append(feedback_b)
                    
                
                self.feedback_vars_bundled.append(feedback_var)
                self.feedback_vars_bundled.append(None)#bias feedback -- not implemented
                
                #'''

                #self.feedback_vars_bundled.append(nn.Parameter(torch.zeros(1)))#weight feedback -- not implemented
                #self.feedback_vars_bundled.append(nn.Parameter(torch.zeros(1)))#bias feedback -- not implemented
                

            elif name is 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))
                
                self.vars_plasticity.append(nn.Parameter(torch.ones(*param[:4])))
                self.vars_plasticity.append(nn.Parameter(torch.ones(param[1])))
                #self.activations_list.append([])
                self.plasticity.append(nn.Parameter(torch.zeros(1))) #not implemented
                self.neuron_plasticity.append(nn.Parameter(torch.zeros(1))) #not implemented
                self.layer_plasticity.append(nn.Parameter(torch.zeros(1))) #not implemented


                self.feedback_vars_bundled.append(nn.Parameter(torch.zeros(1)))#weight feedback -- not implemented
                self.feedback_vars_bundled.append(nn.Parameter(torch.zeros(1)))#bias feedback -- not implemented
                
                
            elif name is 'linear':

                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                
                self.vars_plasticity.append(nn.Parameter(torch.ones(*param)))
                self.vars_plasticity.append(nn.Parameter(torch.ones(param[0])))
                #self.activations_list.append([])
                self.plasticity.append(nn.Parameter(self.init_plasticity * torch.ones(*param)))
                self.neuron_plasticity.append(nn.Parameter(self.init_plasticity * torch.ones(param[0])))
                self.layer_plasticity.append(nn.Parameter(self.init_plasticity * torch.ones(1)))


                feedback_var = []
                
                for fl in range(num_feedback_layers):
                    in_dim = self.width
                    out_dim = self.width
                    if fl == num_feedback_layers - 1:
                        out_dim = param[0]
                    if fl == 0:
                        in_dim = num_outputs
                    feedback_w_shape = [out_dim, in_dim]
                    feedback_w = nn.Parameter(torch.ones(feedback_w_shape))
                    feedback_b =  nn.Parameter(torch.zeros(out_dim))
                    torch.nn.init.kaiming_normal_(feedback_w)
                    feedback_var.append((feedback_w, feedback_b))
                    self.feedback_vars.append(feedback_w)
                    self.feedback_vars.append(feedback_b)
                self.feedback_vars_bundled.append(feedback_var)
                self.feedback_vars_bundled.append(None)#bias feedback -- not implemented

            elif name is 'cat':
                pass
            elif name is 'cat_start':
                pass
            elif name is "rep":
                pass
            elif name is 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                self.vars_plasticity.append(nn.Parameter(torch.ones(param[0])))
                self.vars_plasticity.append(nn.Parameter(torch.ones(param[0])))
                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])

            elif name in ['tanh', 'relu', 'leakyrelu', 'sigmoid', 'linear_act']:
              self.activations_list.append([])
              self.feedback_strength_vars.append(nn.Parameter(self.init_feedback_strength * torch.ones(1)))

            elif name in ['upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape']:
                continue
            else:
                raise NotImplementedError
       

    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)' % (param[1], param[0])
                info += tmp + '\n'

            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)' % (param[0])
                info += tmp + '\n'

            elif name is 'cat':
                tmp = 'cat'
                info += tmp + "\n"
            elif name is 'cat_start':
                tmp = 'cat_start'
                info += tmp + "\n"

            elif name is 'rep':
                tmp = 'rep'
                info += tmp + "\n"


            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info

    def forward(self, x, vars=None, bn_training=True, feature=False):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """
        cat_var = False
        cat_list = []

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0
        
        layer = 0
        
        self.activations_list[layer] = x#.clone()
        
        layer += 1

        for name, param in self.config:
            # assert(name == "conv2d")
            if name == 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                #print('old x shape', x.size())
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                #print('new x shape', x.size())
                idx += 2
                
                #self.activations_list[layer] = x.clone()
                
                #layer += 1

                # print(name, param, '\tout:', x.shape)
            elif name == 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                
                #self.activations_list[layer] = x.clone()
                #layer += 1

            elif name == 'linear':

                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)

                if cat_var:
                    cat_list.append(x)
                idx += 2
            
                #self.activations_list[layer] = x.clone()
                #layer += 1

            elif name == 'rep':
                # print(x.shape)
                if feature:
                    return x
            elif name == "cat_start":
                cat_var = True
                cat_list = []

            elif name == "cat":
                cat_var = False
                x = torch.cat(cat_list, dim=1)

            elif name == 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2
            elif name == 'flatten':
                # print(x.shape)

                x = x.view(x.size(0), -1)

            elif name == 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name == 'relu':
                x = F.relu(x, inplace=param[0])
                self.activations_list[layer] = x#.clone();
                layer += 1
            elif name == 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
                self.activations_list[layer] = x#.clone();
                layer += 1
            elif name == 'tanh':
                x = F.tanh(x)
                self.activations_list[layer] = x#.clone();
                layer += 1
            elif name == 'sigmoid':
                x = torch.sigmoid(x)
                self.activations_list[layer] = x#.clone();
                layer += 1
            elif name == 'linear_act':
                self.activations_list[layer] = x#.clone();
                layer += 1
            elif name == 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name == 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name == 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])

            else:
                raise NotImplementedError
    
    

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)

        return x

    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars

    def getOjaUpdate(self, ground_truth, out, vars, hebbian=False):
      
      #print('ipm', self.inner_plasticity_multiplier)
      idx = 0
      layer = 0
      #print('update')
      num_layers = len(self.activations_list)
      new_vars = []
        
      y_onehot = torch.FloatTensor(out.shape[0], out.shape[1])
      y_onehot = y_onehot.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
      y_onehot.zero_()
      y_onehot.scatter_(1, ground_truth.view(-1, 1), 1)
      loss = 0.5*torch.sum((out - (y_onehot+out.detach()))**2)
      if self.use_error:
            #print('gt', ground_truth)
            #print('out', out)
            loss = torch.nn.functional.cross_entropy(out, ground_truth)
      
      #loss = torch.nn.functional.cross_entropy(out, ground_truth)
      if not self.use_derivative:
          grad = torch.autograd.grad(loss, out, retain_graph=True)[0]#.detach()#-y_onehot#
      else:
          grad = y_onehot #dummy, gets replaced
      #print('grad', grad.size(), grad)
      #print(grad)
      if vars is None:
        vars = self.vars
      
      for var in vars:
        new_vars.append(var)
      
      for name, param in self.config:
        #print(name, idx, layer)
        if name == 'conv2d':
        
          #if not hasattr(vars[idx], 'learn'):
          #  vars[idx].learn = True
            
          if not hasattr(self, 'optimize_out'):
            self.optimize_out = False
            
          if not hasattr(self, 'use_error'):
            self.use_error = False 
          
          if not hasattr(self, 'linear_feedback'):
            self.linear_feedback = False 
           
          if not hasattr(self, 'use_derivative'):
            self.use_derivative = False 
            
          if not hasattr(self, 'error_only_to_output'):
            self.error_only_to_output = False 
           
          if not hasattr(self, 'neuron_level_plasticity'):
            self.neuron_level_plasticity = False 
            
           
          if not hasattr(self, 'layer_level_plasticity'):
            self.layer_level_plasticity = False 
            
          if vars[idx].learn:
        
              w, b = vars[idx], vars[idx + 1]

              y_onehot = torch.FloatTensor(out.shape[0], out.shape[1])
              y_onehot = y_onehot.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
              y_onehot.zero_()
              y_onehot.scatter_(1, ground_truth.view(-1, 1), 1)

              if layer == num_layers - 1 and not self.optimize_out:
                  if self.use_error:
                    next_activations = -grad
                    
                  else:
                    next_activations = y_onehot
              else:
                
                feedback_var = self.feedback_vars_bundled[idx]
                if self.error_only_to_output:
                    if self.use_error:
                        next_activations = -grad
                    else:
                        next_activations = y_onehot
                        
                    next_activations = torch.abs(next_activations)
                else:
                    if self.use_error:
                        next_activations = -grad
                    else:
                        next_activations = y_onehot
                for fl in range(self.num_feedback_layers):
                    feedback_w = feedback_var[fl][0]
                    feedback_b = feedback_var[fl][1]
                    next_activations = F.linear(next_activations, feedback_w, feedback_b)
                    if not self.linear_feedback:
                        next_activations = F.relu(next_activations, inplace=param[0])
                        
                    #if self.use_derivative:
                    #    if layer == num_layers-1:
                    #        next_activations = next_activations * torch.sign(out)
                    #    else:
                    #        next_activations = next_activations * torch.sign(self.activations_list[layer+1])

                #print('activations size', next_activations.size(), self.activations_list[layer+1].size())
                activations = self.activations_list[layer]
                next_activations = next_activations.view(self.activations_list[layer+1].size())

                #print(yoo)
                if layer == num_layers - 1:
                    next_activations = self.feedback_strength_vars[layer+1] * next_activations + (torch.ones(1).cuda() - self.feedback_strength_vars[layer+1]) * out
                    if self.use_derivative:
                        next_activations = -torch.autograd.grad(loss, out)[0]#.detach()
                else:    
                    next_activations = self.feedback_strength_vars[layer+1] * next_activations + (torch.ones(1).cuda() - self.feedback_strength_vars[layer+1]) * self.activations_list[layer+1]
                    if self.use_derivative:
                        next_activations = -torch.autograd.grad(loss, self.activations_list[layer+1])[0]* (1+torch.sign(self.activations_list[layer+1]))*0.5#.detach()

                #print('activations size', activations.size(), 'next activations size', next_activations.size())
                
                #next_activations = torch.transpose(next_activations, 0, 1)
                
                
                new_activations = torch.Tensor(next_activations.size(0), next_activations.size(2), next_activations.size(3), activations.size(1), param[3], param[3]).cuda()
                
                divide_factor = next_activations.size(2) * next_activations.size(3)
                
                stride = param[4]
                padding = param[5]
                newcoord = -1
                for xcoord in range(0, activations.shape[2]-param[3]+1, stride):
                    newcoord += 1
                    #print(newcoord, xcoord, activations.shape[2]-stride+1, activations.shape[2], stride)
                    new_activations[:, newcoord, newcoord, :, :, :] = activations[:, :, xcoord:xcoord+param[3], xcoord:xcoord+param[3]]
                    
                next_activations = torch.transpose(next_activations, 1, 2)
                next_activations = torch.transpose(next_activations, 2, 3)
                next_activations = next_activations.contiguous().view(-1, next_activations.size(3))
                
                new_activations = new_activations.view(new_activations.size(0)*new_activations.size(1)*new_activations.size(2), -1)
                
                activations = new_activations
                #print('df', divide_factor)
                next_activations = next_activations / divide_factor
                activations = activations / divide_factor
                
                activations = torch.clamp(activations, 0.0, 1.0)
                next_activations = torch.clamp(next_activations, 0.0, 1.0)
                activations[torch.isnan(activations)] = 0.0
                next_activations[torch.isnan(next_activations)] = 0.0
                #print('act mag', torch.mean(torch.abs(activations)), torch.mean(torch.abs(next_activations)))
                #print('activations size', activations.size(), 'next activations size', next_activations.size())
                        
              #if (len(activations.shape) > 2):
              #  activations = activations.view(next_activations.shape[0], -1)

              #print(self.plasticity[layer].shape, next_activations.shape, activations.shape, w.shape)

              if hebbian:
                if self.neuron_level_plasticity:
                    oja_update = self.neuron_plasticity[layer].view(-1, 1) * (torch.mm(torch.t(next_activations), activations))
                elif self.layer_level_plasticity:
                    oja_update = self.layer_plasticity[layer] * (torch.mm(torch.t(next_activations), activations))
                else:
                    oja_update = self.plasticity[layer] * (torch.mm(torch.t(next_activations), activations))
              else:
                if next_activations.size()[0] == 1:
                    if self.neuron_level_plasticity:
                         oja_update = self.neuron_plasticity[layer].view(-1, 1) * (torch.mm(torch.t(next_activations), activations) - (torch.t(next_activations)**2) * w)
                    elif self.layer_level_plasticity:
                         oja_update = self.layer_plasticity[layer] * (torch.mm(torch.t(next_activations), activations) - (torch.t(next_activations)**2) * w)
                    else:
                        oja_update = self.plasticity[layer] * (torch.mm(torch.t(next_activations), activations) - (torch.t(next_activations)**2) * w)
                else:
                    if self.neuron_level_plasticity:
                        oja_update = self.neuron_plasticity[layer].view(-1, 1) * (torch.mm(torch.t(next_activations), activations) - torch.sum((next_activations**2), 0).unsqueeze(1) * w)
                    elif self.layer_level_plasticity:
                        oja_update = self.layer_plasticity[layer] * (torch.mm(torch.t(next_activations), activations) - torch.sum((next_activations**2), 0).unsqueeze(1) * w)
                    else:
                        #print('plast size', self.plasticity[layer].size())
                        #print('w size', w.size())
                        oja_update = self.plasticity[layer] * (torch.mm(torch.t(next_activations), activations) - torch.sum((next_activations**2), 0).unsqueeze(1) * w.view(w.size(0), w.size(1)*w.size(2)*w.size(3)))

              #if layer == num_layers - 1:
              #print('oja mag', torch.sum(torch.abs(oja_update)))
              new_vars[idx] = new_vars[idx] + oja_update.view(w.size(0), w.size(1), w.size(2), w.size(3))
              new_vars[idx].learn = vars[idx].learn
              #print("Activations shape:  ", activations.shape)
              #print("Out shape:  ", out.shape)
              #print("ground_truth shape:  ", ground_truth.shape)
              #sys.exit()
          idx += 2
          layer += 1

        elif name == 'convt2d':
          #have not implemented hebbian update for conv.  would need to think about how to do it
        
          w, b = vars[idx], vars[idx + 1]
          activations = self.activations_list[layer]
          idx += 2
          layer += 1

        elif name == 'linear':
            
          #if not hasattr(vars[idx], 'learn'):
          #  vars[idx].learn = True
            
          if not hasattr(self, 'optimize_out'):
            self.optimize_out = False
            
          if not hasattr(self, 'use_error'):
            self.use_error = False 
          
          if not hasattr(self, 'linear_feedback'):
            self.linear_feedback = False 
           
          if not hasattr(self, 'use_derivative'):
            self.use_derivative = False 
            
          if not hasattr(self, 'error_only_to_output'):
            self.error_only_to_output = False 
           
          if not hasattr(self, 'neuron_level_plasticity'):
            self.neuron_level_plasticity = False 
            
           
          if not hasattr(self, 'layer_level_plasticity'):
            self.layer_level_plasticity = False 
            
          if not hasattr(self, 'inner_plasticity_multiplier'):
            self.inner_plasticity_multiplier = 1 
            
            
          if vars[idx].learn:
              #print('grad', grad)
              w, b = vars[idx], vars[idx + 1]

              y_onehot = torch.FloatTensor(out.shape[0], out.shape[1])
              y_onehot = y_onehot.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
              y_onehot.zero_()
              y_onehot.scatter_(1, ground_truth.view(-1, 1), 1)

              if layer == num_layers - 1 and not self.optimize_out:
                  if self.use_error:
                    next_activations = -grad#y_onehot - (out / torch.sum(out, 1, keepdim=True))#
                  else:
                    next_activations = y_onehot
              else:
                
                feedback_var = self.feedback_vars_bundled[idx]
                if self.error_only_to_output:
                    if self.use_error:
                        next_activations = -grad#y_onehot - (out / torch.sum(out, 1, keepdim=True))#-grad
                    else:
                        next_activations = y_onehot
                        
                    next_activations = torch.abs(next_activations)
                else:
                    if self.use_error:
                        next_activations = -grad#y_onehot - (out / torch.sum(out, 1, keepdim=True))#-grad
                    else:
                        next_activations = y_onehot
                        
                #print('nawayb4', next_activations)
                for fl in range(self.num_feedback_layers):
                    feedback_w = feedback_var[fl][0]
                    feedback_b = feedback_var[fl][1]
                    next_activations = F.linear(next_activations, feedback_w, feedback_b)
                    #print('fb_w', torch.sum(torch.abs(feedback_w)), torch.mean(feedback_w), dim=1)
                    #print('fb_b', torch.sum(torch.abs(feedback_b)), torch.mean(feedback_b), dim=1)
                    #print('nalinear', torch.sum(torch.abs(next_activations)), torch.mean(next_activations))
                    if not self.linear_feedback:
                        next_activations = F.relu(next_activations, inplace=param[0])
                        
                    #if self.use_derivative:
                    #    if layer == num_layers-1:
                    #        next_activations = next_activations * torch.sign(out)
                    #    else:
                    #        next_activations = next_activations * torch.sign(self.activations_list[layer+1])

                if layer == num_layers - 1:
                    next_activations = self.feedback_strength_vars[layer+1] * next_activations + (torch.ones(1).cuda() - self.feedback_strength_vars[layer+1]) * out
                    #next_activations = next_activations.detach()
                    if self.use_derivative:
                        next_activations = -torch.autograd.grad(loss, out)[0]#.detach()
                else:    
                    #print('nab4', torch.mean(torch.abs(next_activations)))
                    next_activations = self.feedback_strength_vars[layer+1] * next_activations + (torch.ones(1).cuda() - self.feedback_strength_vars[layer+1]) * self.activations_list[layer+1]
                    #next_activations = next_activations.detach()
                    #print('naafter', torch.mean(torch.abs(next_activations)))
                    #print('before', torch.sum(next_activations**2))
                    if self.use_derivative:
                        next_activations = -torch.autograd.grad(loss, self.activations_list[layer+1])[0]* (1+torch.sign(self.activations_list[layer+1]))*0.5#.detach()
                    #print('naafterderiv', torch.mean(torch.abs(next_activations)))
                        #print('after', torch.sum(next_activations**2))

              activations = self.activations_list[layer]
              if (len(activations.shape) > 2):
                activations = activations.view(next_activations.shape[0], -1)

              #print(self.plasticity[layer].shape, next_activations.shape, activations.shape, w.shape)

              if hebbian:
                if self.neuron_level_plasticity:
                    oja_update = self.neuron_plasticity[layer].view(-1, 1) * (torch.mm(torch.t(next_activations), activations))
                elif self.layer_level_plasticity:
                    oja_update = self.layer_plasticity[layer] * (torch.mm(torch.t(next_activations), activations))
                else:
                    oja_update = self.plasticity[layer] * (torch.mm(torch.t(next_activations), activations))
              else:
                if next_activations.size()[0] == 1:
                    if self.neuron_level_plasticity:
                         oja_update = self.neuron_plasticity[layer].view(-1, 1) * (torch.mm(torch.t(next_activations), activations) - (torch.t(next_activations)**2) * w)
                    elif self.layer_level_plasticity:
                         oja_update = self.layer_plasticity[layer] * (torch.mm(torch.t(next_activations), activations) - (torch.t(next_activations)**2) * w)
                    else:
                        oja_update = self.plasticity[layer] * (torch.mm(torch.t(next_activations), activations) - (torch.t(next_activations)**2) * w)
                else:
                    if self.neuron_level_plasticity:
                        oja_update = self.neuron_plasticity[layer].view(-1, 1) * (torch.mm(torch.t(next_activations), activations) - torch.sum((next_activations**2), 0).unsqueeze(1) * w)
                    elif self.layer_level_plasticity:
                        oja_update = self.layer_plasticity[layer] * (torch.mm(torch.t(next_activations), activations) - torch.sum((next_activations**2), 0).unsqueeze(1) * w)
                    else:
                        oja_update = self.plasticity[layer] * (torch.mm(torch.t(next_activations), activations) - torch.sum((next_activations**2), 0).unsqueeze(1) * w)
                        
              if layer != num_layers - 1:
                oja_update = oja_update * self.inner_plasticity_multiplier


              #print("layer", layer)
              #print('change size', oja_update.size())
              #print('change abs', torch.mean(torch.abs(oja_update)))
              #print('change mean', torch.mean(oja_update))
              #print('vars abs', torch.mean(torch.abs(new_vars[idx])))
              #print('vars mean', torch.mean(new_vars[idx]))
              #if layer == num_layers - 1:
              
              new_vars[idx] = new_vars[idx] + oja_update
              new_vars[idx].learn = vars[idx].learn
              #print("Activations shape:  ", activations.shape)
              #print("Out shape:  ", out.shape)
              #print("ground_truth shape:  ", ground_truth.shape)
              #sys.exit()
          idx += 2
          layer += 1

        elif name == 'bn':
          idx += 2
          #do not increment "layer"!
      #for nv in range(len(new_vars)):
      #  new_vars[nv] = torch.nn.Parameter(new_vars[nv])
      #  new_vars[nv].learn = vars[nv].learn
      return new_vars;

