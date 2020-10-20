import numpy as np

class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_type, dataset, in_channels=6, num_actions=6, width=300, num_extra_dense_layers=0):

        if "Sin" == dataset:
            if model_type=="old":
                hidden_size = width
                return [
                    ('linear', [hidden_size, in_channels]),
                    ('relu', [True]),
                    ('linear', [hidden_size, hidden_size]),
                    ('relu', [True]),
                    ('linear', [hidden_size, hidden_size]),
                    ('relu', [True]),
                    ('linear', [hidden_size, hidden_size]),
                    ('relu', [True]),
                    ('linear', [hidden_size, hidden_size]),
                    ('relu', [True]),
                    ('linear', [hidden_size * 3, hidden_size]),
                    ('relu', [True]),
                    ('rep', []),
                    ('linear', [hidden_size, hidden_size * 3]),
                    ('relu', [True]),
                    ('linear', [hidden_size, hidden_size]),
                    ('relu', [True]),
                    ('linear', [num_actions, hidden_size]),
                    ('linear_act', [True])
                ]
            elif model_type=="short3wide":
                big_hidden_size = int(width * np.sqrt(11))
                return [
                    ('linear', [big_hidden_size, in_channels]),
                    ('relu', [True]),
                    ('linear', [big_hidden_size, big_hidden_size]),
                    ('relu', [True]),
                    ('linear', [num_actions, big_hidden_size]),
                    ('linear_act', [True])
                ]
            elif model_type=="short3wide5":
                big_hidden_size = int(width * 5)
                return [
                    ('linear', [big_hidden_size, in_channels]),
                    ('relu', [True]),
                    ('linear', [big_hidden_size, big_hidden_size]),
                    ('relu', [True]),
                    ('linear', [num_actions, big_hidden_size]),
                    ('linear_act', [True])
                ]   
            elif model_type=="short3expand11":
                #big_hidden_size = int(width * 4)
                return [
                    ('linear', [width*11, in_channels]),
                    ('relu', [True]),
                    ('linear', [width, width*11]),
                    ('relu', [True]),
                    ('linear', [num_actions, width]),
                    ('linear_act', [True])
                ]  
            elif model_type=="short3":
                hidden_size = width
                return [
                    ('linear', [hidden_size, in_channels]),
                    ('relu', [True]),
                    ('linear', [hidden_size, hidden_size]),
                    ('relu', [True]),
                    ('linear', [num_actions, hidden_size]),
                    ('linear_act', [True])
                ]
            elif model_type=="med3":
                hidden_size = width
                return [
                    ('linear', [hidden_size, in_channels]),
                    ('relu', [True]),
                    ('linear', [hidden_size, hidden_size]),
                    ('relu', [True]),
                    ('linear', [hidden_size, hidden_size]),
                    ('relu', [True]),
                    ('linear', [num_actions, hidden_size]),
                    ('linear_act', [True])
                ]
            
            elif model_type=="med3wide3":
                hidden_size = width
                return [
                    ('linear', [hidden_size, in_channels]),
                    ('relu', [True]),
                    ('linear', [3*hidden_size, hidden_size]),
                    ('relu', [True]),
                    ('linear', [hidden_size, 3*hidden_size]),
                    ('relu', [True]),
                    ('linear', [num_actions, hidden_size]),
                    ('linear_act', [True])
                ]
            
            elif model_type=="short1widesuper":
                hidden_size = width
                return [
                    ('linear', [hidden_size*hidden_size*16, in_channels]),
                    ('relu', [True]),
                    ('linear', [num_actions, hidden_size*hidden_size*16]),
                    ('linear_act', [True])
                ]
            
            elif model_type=="linear":
                hidden_size = width
                return [
                    ('linear', [hidden_size, in_channels]),
                    ('relu', [True]),
                    ('linear', [hidden_size, hidden_size]),
                    ('relu', [True]),
                    ('linear', [hidden_size, hidden_size]),
                    ('relu', [True]),
                    ('linear', [hidden_size * 5, hidden_size]),
                    ('relu', [True]),
                    ('rep', []),
                    ('linear', [num_actions, hidden_size * 5]),
                    ('linear_act', [True])
                ]

            elif model_type=="non-linear":
                hidden_size = width
                return [
                    ('linear', [hidden_size, in_channels]),
                    ('relu', [True]),
                    ('linear', [hidden_size, hidden_size]),
                    ('relu', [True]),
                    ('linear', [hidden_size, hidden_size]),
                    ('relu', [True]),
                    ('linear', [hidden_size * 5, hidden_size]),
                    ('relu', [True]),
                    ('rep', []),
                    ('linear', [hidden_size, hidden_size * 5]),
                    ('relu', [True]),
                    ('linear', [num_actions, hidden_size])
                    ('linear_act', [True])

                ]

        elif dataset == "omniglot":
          if model_type=="eigthsize":
            channels = 256
            # channels = 256
            layers =  [
                ('conv2d', [channels, 1, 3, 3, 2, 0]),
                ('relu', [True]),
                # ('bn', [64]),
                ('conv2d', [channels, channels, 3, 3, 1, 0]),
                ('relu', [True]),

                ('conv2d', [channels, channels, 3, 3, 2, 0]),
                ('relu', [True]),

                ('conv2d', [channels, channels, 3, 3, 1, 0]),
                ('relu', [True]),
                # ('bn', [128]),
                ('conv2d', [channels, channels, 3, 3, 2, 0]),
                ('relu', [True]),
                # ('bn', [256]),
                ('conv2d', [channels, channels, 3, 3, 2, 0]),
                ('relu', [True]),
                # ('bn', [512]),
                ('flatten', []),
                ('rep', []),

                ('linear', [1024, 9 * channels]),
                ('relu', [True])
            ]
            for lay in range(num_extra_dense_layers):
                layers.append(('linear', [1024, 1024]))
                layers.append(('relu', [True]))
            
            layers.append(('linear', [1000, 1024]))
            return layers
          elif model_type=="halfsize":
            channels = 128
            # channels = 256
            layers =  [
                ('conv2d', [channels, 1, 3, 3, 2, 0]),
                ('relu', [True]),
                # ('bn', [64]),
                ('conv2d', [channels, channels, 3, 3, 1, 0]),
                ('relu', [True]),

                ('conv2d', [channels, channels, 3, 3, 2, 0]),
                ('relu', [True]),

                ('conv2d', [channels, channels, 3, 3, 1, 0]),
                ('relu', [True]),
                # ('bn', [128]),
                ('conv2d', [channels, channels, 3, 3, 2, 0]),
                ('relu', [True]),
                # ('bn', [256]),
                ('conv2d', [channels, channels, 3, 3, 2, 0]),
                ('relu', [True]),
                # ('bn', [512]),
                ('flatten', []),
                ('rep', []),

                ('linear', [128, 9 * channels]),
                ('relu', [True])
            ]
            for lay in range(num_extra_dense_layers):
                layers.append(('linear', [128, 128]))
                layers.append(('relu', [True]))
            
            layers.append(('linear', [1000, 128]))
            return layers
        
          else:
            channels = 256
            # channels = 256
            layers =  [
                ('conv2d', [channels, 1, 3, 3, 2, 0]),
                ('relu', [True]),
                # ('bn', [64]),
                ('conv2d', [channels, channels, 3, 3, 1, 0]),
                ('relu', [True]),

                ('conv2d', [channels, channels, 3, 3, 2, 0]),
                ('relu', [True]),

                ('conv2d', [channels, channels, 3, 3, 1, 0]),
                ('relu', [True]),
                # ('bn', [128]),
                ('conv2d', [channels, channels, 3, 3, 2, 0]),
                ('relu', [True]),
                # ('bn', [256]),
                ('conv2d', [channels, channels, 3, 3, 2, 0]),
                ('relu', [True]),
                # ('bn', [512]),
                ('flatten', []),
                ('rep', []),

                ('linear', [width, 9 * channels]),
                ('relu', [True])
            ]
            for lay in range(num_extra_dense_layers):
                layers.append(('linear', [width, width]))
                layers.append(('relu', [True]))
            
            layers.append(('linear', [1000, width]))
            return layers

        elif dataset == "imagenet":
            channels = 256
            # channels = 256
            layers = [
                ('conv2d', [channels, 3, 3, 3, 2, 0]),
                ('relu', [True]),
                # ('bn', [64]),
                ('conv2d', [channels, channels, 3, 3, 1, 0]),
                ('relu', [True]),

                ('conv2d', [channels, channels, 3, 3, 2, 0]),
                ('relu', [True]),

                ('conv2d', [channels, channels, 3, 3, 1, 0]),
                ('relu', [True]),
                # ('bn', [128]),
                ('conv2d', [channels, channels, 3, 3, 2, 0]),
                ('relu', [True]),
                # ('bn', [256]),
                ('conv2d', [channels, channels, 3, 3, 2, 0]),
                ('relu', [True]),
                # ('bn', [512]),
                ('flatten', []),
                ('rep', []),

                ('linear', [width, 9 * channels]),
                ('relu', [True])
            ]
            for lay in range(num_extra_dense_layers):
                layers.append(('linear', [width, width]))
                layers.append(('relu', [True]))
            
            layers.append(('linear', [1000, width]))
            return layers
        

        else:
            print("Unsupported model; either implement the model in model/ModelFactory or choose a different model")
            assert (False)
