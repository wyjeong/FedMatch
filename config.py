def set_config(args):
    """ Model, Data, and Training Coniguration

    Specifies detailed configurations, such as batch-size, number of epcohs and rounds, 
    hyper-parameters, etc., can be set in this file.

    Created by: 
        Wonyong Jeong (wyjeong@kaist.ac.kr)
    """

    args.gpu_mem = 7 # Gbyte (adjust this as needed)
    args.dataset_path = '/path/to/data/'  # for datasets
    args.output_path =   '/path/to/outputs/' # for logs, weights, etc.
    
    args.archi = 'resnet9'
    args.dataset_id_to_name = {0: 'cifar_10'}
    
    # scenarios
    if 'lc' in args.task:
        args.scenario = 'labels-at-client'
        args.num_labels_per_class = 5
        args.num_epochs_client = 1 
        args.batch_size_client = 10 # for labeled set
        args.num_epochs_server = 0
        args.batch_size_server = 0
        args.num_epochs_server_pretrain = 0
        args.lr_factor = 3
        args.lr_patience = 5
        args.lr_min = 1e-20
    elif 'ls' in args.task:
        args.scenario = 'labels-at-server'
        args.num_labels_per_class = 100
        args.num_epochs_client = 1 
        args.batch_size_client = 100
        args.batch_size_server = 100
        args.num_epochs_server = 1
        args.num_epochs_server_pretrain = 1
        args.lr_factor = 3
        args.lr_patience = 20
        args.lr_min = 1e-20

    # tasks
    if 'biid' in args.task or 'bimb'in args.task: 
        args.sync = False
        args.num_tasks = 1
        args.num_clients = 100
        args.num_rounds = 200 
    
    # datasets
    if 'c10' in args.task:
        args.dataset_id = 0
        args.num_classes = 10
        args.num_test = 2000
        args.num_valid = 2000
        args.batch_size_test = 100

    # base networks
    if args.archi in ['resnet9']:
        args.lr = 1e-3
        args.wd = 1e-4

    # hyper-parameters
    if args.model in ['fedmatch']:
        args.num_helpers = 2
        args.confidence = 0.75
        args.psi_factor = 0.2
        args.h_interval = 10

        if args.scenario == 'labels-at-client':
            args.lambda_s = 10 # supervised learning
            args.lambda_i = 1e-2 # inter-client consistency
            args.lambda_a = 1e-2 # agreement-based pseudo labeling
            args.lambda_l2 = 10
            args.lambda_l1 = 1e-4
            args.l1_thres = 1e-6 * 5
            args.delta_thres = 1e-5 * 5
                
        elif args.scenario == 'labels-at-server':
            args.lambda_s = 10 # supervised learning
            args.lambda_i = 1e-2 # inter-client consistency
            args.lambda_a = 1e-2 # agreement-based pseudo labeling
            args.lambda_l2 = 10
            args.lambda_l1 = 1e-5
            args.l1_thres = 1e-5
            args.delta_thres = 1e-5 

    return args


