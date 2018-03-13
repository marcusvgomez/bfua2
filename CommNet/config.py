"""
Training config

Contains runtime config params
"""

class RuntimeConfig:

    def __init__(self, args):
        print (args)
        self.use_cuda = args['use_cuda']
        self.n_epochs = 10000 if 'n-epochs' not in args else args['n-epochs']
        self.load_weights = True if 'load-model' in args else False
        self.load_weights_path = "" if 'load-model' not in args else args['load-model']
        self.save_weights = True if 'save-model' in args else False
        self.save_model_epoch = float("inf") if 'save-model-epoch' not in args else args['save-model-epoch']
        self.save_weights_path = "" if 'save-model' not in args else args['save-model']
        
        #input network parameters
        self.num_agents = 3 if 'num-agents' not in args else args['num-agents']
        self.hidden_size = 1024 if 'hidden-size' not in args else args['hidden-size'] 
        self.K = 20 if 'comm-steps' not in args else args['comm-steps']

        #preset network parameters these are somewhat hard coded into agent
        self.minibatch_size = 1024 if 'minibatch-size' not in args else args['minibatch-size']
        self.num_gpus = 2


        #hyperparameters
        self.learning_rate = 0.005 if 'learning-rate' not in args else args['learning-rate']
        self.optimizer_decay = float("-inf") if 'optimizer-decay-epoch' not in args else args['optimizer-decay']
        self.optimizer_decay_rate = 5 if 'optimizer-decay-rate' not in args else args['optimizer-decay-rate']
        self.dropout = 0 if 'dropout' not in args else args['dropout']

        self.env = "Levers" if 'env' not in args else args['env']
        self.is_threaded = True if 'is_threaded' not in args else args['is_threaded']
