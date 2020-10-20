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
            
            
    def __init__(self, config, num_feedback_layers, init_plasticity, init_feedback_strength, width=1024, feedback_l2=0.0, optimize_out=False, use_error=False, feedback_only_to_output=False, error_only_to_output=False, linear_feedback=False, use_derivative=False, neuron_level_plasticity=False, layer_level_plasticity=False, coarse_level_plasticity=False):
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
        self.feedback_only_to_output = feedback_only_to_output   
        self.error_only_to_output = error_only_to_output   
        self.linear_feedback = linear_feedback
        self.use_derivative = use_derivative
        
        self.neuron_level_plasticity = neuron_level_plasticity
        
        self.layer_level_plasticity = layer_level_plasticity

        self.coarse_level_plasticity = coarse_level_plasticity


        
        num_outputs = self.config[-1][1][0]

        self.plasticity = nn.ParameterList()
        
        self.neuron_plasticity = nn.ParameterList()
        
        self.layer_plasticity = nn.ParameterList()

        self.coarse_plasticity = nn.Parameter(torch.zeros([1]))


        
        for i, (name, param) in enumerate(self.config):
            if name is 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                #self.activations_list.append([])
                self.plasticity.append(nn.Parameter(torch.zeros(1))) #not implemented
                self.neuron_plasticity.append(nn.Parameter(torch.zeros(1))) #not implemented
                self.layer_plasticity.append(nn.Parameter(torch.zeros(1))) #not implemented


                self.feedback_vars_bundled.append(nn.Parameter(torch.zeros(1)))#weight feedback -- not implemented
                self.feedback_vars_bundled.append(nn.Parameter(torch.zeros(1)))#bias feedback -- not implemented
                

            elif name is 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))
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
        
        self.activations_list[layer] = x.clone()
        #print('input_shape', x.size())
        layer += 1

        for name, param in self.config:
            # assert(name == "conv2d")
            if name == 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
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
                self.activations_list[layer] = x.clone();
                layer += 1
            elif name == 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
                self.activations_list[layer] = x.clone();
                layer += 1
            elif name == 'tanh':
                x = F.tanh(x)
                self.activations_list[layer] = x.clone();
                layer += 1
            elif name == 'sigmoid':
                x = torch.sigmoid(x)
                self.activations_list[layer] = x.clone();
                layer += 1
            elif name == 'linear_act':
                self.activations_list[layer] = x.clone();
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
      idx = 0
      layer = 0
      #print('update')
      num_layers = len(self.activations_list)
      new_vars = []
      if vars is None:
        vars = self.vars
      
      for var in vars:
        new_vars.append(var)
      
      for name, param in self.config:
        #print(name, idx, layer)
        if name == 'conv2d':
        
          #have not implemented hebbian update for conv.  would need to think about how to do it
          w, b = vars[idx], vars[idx + 1]
          activations = self.activations_list[layer]
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
            
            
          if not hasattr(self, 'use_error'):
            self.use_error = False 
            
          if not hasattr(self, 'feedback_only_to_output'):
            self.feedback_only_to_output = False 
            
          if not hasattr(self, 'error_only_to_output'):
            self.error_only_to_output = False 
            
          if not hasattr(self, 'linear_feedback'):
            self.linear_feedback = False 
           
          if not hasattr(self, 'use_derivative'):
            self.use_derivative = False 
            
          if not hasattr(self, 'neuron_level_plasticity'):
            self.neuron_level_plasticity = False 
            
          if not hasattr(self, 'layer_level_plasticity'):
            self.layer_level_plasticity = False 
          
          if not hasattr(self, 'coarse_level_plasticity'):
            self.coarse_level_plasticity = False 
            
          #print('what', 'idx', idx, 'layer', layer, self.activations_list[layer].size())
          if vars[idx].learn:
        
              w, b = vars[idx], vars[idx + 1]




              feedback_var = self.feedback_vars_bundled[idx]
                
              if self.error_only_to_output:
                if layer == num_layers - 1:
                  if self.use_error:
                    next_activations = ground_truth - out
                  else:
                    next_activations = ground_truth
                else:
                  if self.use_error:
                    next_activations = ground_truth - out
                  else:
                    next_activations = ground_truth
                    
                  next_activations = torch.abs(next_activations)
              else:
                if self.use_error:
                  next_activations = ground_truth - out
                else:
                  next_activations = ground_truth

              #print('yooo', ground_truth.size(), ground_truth)
              for fl in range(self.num_feedback_layers):
                #print('fl', fl, next_activations.size(), feedback_var[fl][0].size(), feedback_var[fl][1].size())
                feedback_w = feedback_var[fl][0]
                feedback_b = feedback_var[fl][1]
                next_activations = F.linear(next_activations, feedback_w, feedback_b)
                
                if not self.linear_feedback:
                    next_activations = F.relu(next_activations, inplace=param[0])
                    
                if self.use_derivative:
                    if layer == num_layers-1:
                        next_activations = next_activations * torch.sign(out)
                    else:
                        next_activations = next_activations * torch.sign(self.activations_list[layer+1])
                        

              if torch.cuda.is_available():
                  if layer == num_layers - 1:
                    next_activations = self.feedback_strength_vars[layer+1] * next_activations + (torch.ones(1).cuda() - self.feedback_strength_vars[layer+1]) * out
                  else:    
                    if self.feedback_only_to_output:
                        next_activations = 0 * next_activations + (torch.ones(1).cuda() - 0) * self.activations_list[layer+1]
                    else:
                        next_activations = self.feedback_strength_vars[layer+1] * next_activations + (torch.ones(1).cuda() - self.feedback_strength_vars[layer+1]) * self.activations_list[layer+1]

              else:
                  if layer == num_layers - 1:
                    next_activations = self.feedback_strength_vars[layer+1] * next_activations + (torch.ones(1) - self.feedback_strength_vars[layer+1]) * out
                  else:    
                    if self.feedback_only_to_output:
                        next_activations = 0 * next_activations + (torch.ones(1) - 0) * self.activations_list[layer+1]
                    else:
                        next_activations = self.feedback_strength_vars[layer+1] * next_activations + (torch.ones(1) - self.feedback_strength_vars[layer+1]) * self.activations_list[layer+1]


                        
              activations = self.activations_list[layer]
              if (len(activations.shape) > 2):
                activations = activations.view(next_activations.shape[0], -1)

              #print(self.plasticity[layer].shape, next_activations.shape, activations.shape, w.shape)

              #print('yoooooo', self.neuron_plasticity[layer].view(-1, 1))
              if hebbian:
                if self.neuron_level_plasticity:
                    self.neuron_plasticity[layer].view(-1, 1) * (torch.mm(torch.t(next_activations), activations))
                elif self.layer_level_plasticity:
                    self.layer_plasticity[layer] * (torch.mm(torch.t(next_activations), activations))
                elif self.coarse_level_plasticity:
                    self.coarse_plasticity * (torch.mm(torch.t(next_activations), activations))
                else:
                    oja_update = self.plasticity[layer] * (torch.mm(torch.t(next_activations), activations))
              else:
                #print('howdy', self.plasticity[layer].size(), next_activations.size(), activations.size(), w.size())
                a = (torch.mm(torch.t(next_activations), activations))
                #print(a.size())
                b = torch.sum((next_activations**2), 0).unsqueeze(1)
                #print(b.size())
                if self.neuron_level_plasticity:
                    oja_update = self.neuron_plasticity[layer].view(-1, 1) * (torch.mm(torch.t(next_activations), activations) - torch.sum((next_activations**2), 0).unsqueeze(1) * w)
                elif self.layer_level_plasticity:
                    oja_update = self.layer_plasticity[layer] * (torch.mm(torch.t(next_activations), activations) - torch.sum((next_activations**2), 0).unsqueeze(1) * w)
                elif self.coarse_level_plasticity:
                    oja_update = self.coarse_plasticity * (torch.mm(torch.t(next_activations), activations) - torch.sum((next_activations**2), 0).unsqueeze(1) * w)
                else:
                    oja_update = self.plasticity[layer] * (torch.mm(torch.t(next_activations), activations) - torch.sum((next_activations**2), 0).unsqueeze(1) * w)

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


