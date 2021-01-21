import argparse

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=2019,
                            help='Random seed.')

    parser.add_argument('--model', nargs='?', default='kgat',
                            help='Choose a dataset from {kgat}')
    parser.add_argument('--model_type', nargs='?', default='',
                            help='Choose usage of KGATLinear or KGAT.')
    parser.add_argument('--dataset', nargs='?', default='last-fm',
                            help='Choose a dataset from {yelp2018, last-fm, amazon-book}')

    parser.add_argument('--pretrain', type=int, default=1,
                            help='0: No pretrain, 1: Pretrain with the learned embeddings, 2: Pretrain with stored model.')
    parser.add_argument('--pretrained_model', nargs='?', default='',
                            help='Path of stored model.')

    parser.add_argument('--aggr_type', nargs='?', default='Bi',
                            help='Specify the aggregation type of the graph convolutional layer {Bi, GCN, GraphSAGE}.')
    parser.add_argument('--conv_layer_dims', nargs='?', default='[64,32,16]',
                            help='Number of out_channels. Defines the number of layers implicitly.')
    parser.add_argument('--node_dropout', type=float, default=0.1,
                            help='Keep probability w.r.t. node dropout for each layer during training.')
    parser.add_argument('--message_dropout', type=float, default=0.1,
                            help='Keep probability w.r.t. message dropout for each layer during training.')
    parser.add_argument('--negative_slope', type=float, default=0.2,
                            help='Negative Slope for LeakyReLU.')

    parser.add_argument('--entity_user_emb_dim', type=int, default=64,
                            help='Embedding dimension for all nodes (users, entities) in CKG graph.')
    parser.add_argument('--relation_emb_dim', type=int, default=64,
                            help='Embedding dimension for all relations in CKG graph.')

    parser.add_argument('--epochs', type=int, default=1000,
                            help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=0.0001,
                            help='Learning rate.')
    parser.add_argument('--stopping_steps', type=int, default=10,
                            help='Number of epochs for early stopping')

    parser.add_argument('--cf_loss_lambda', type=float, default=1e-5,
                            help='Lambda when calculating CF loss.')
    parser.add_argument('--ckg_loss_lambda', type=float, default=1e-5,
                            help='Lambda when calculating CKG loss.')

    parser.add_argument('--verbose_steps', type=int, default=1,
                            help='Print loss every n-th iteration.')
    parser.add_argument('--evaluate_steps', type=int, default=10,
                            help='Evaluate model every n-th iteration.')

    parser.add_argument('--cf_batch_size', type=int, default=1024,
                            help='Batch size for CF loss.')
    parser.add_argument('--ckg_batch_size', type=int, default=2048,
                            help='Batch size for CKG loss.')
    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
                            help='Output sizes of every layer.')

    parser.add_argument('--save_flag', type=int, default=1,
                            help='0: Do not save model, 1: Save model')

    args = parser.parse_args()
    
    save_dir = 'trained_model/KGAT/{}/{}_entityuserembdim{}_relationembdim{}_{}_lr{}_pretrain{}/'.format(
            args.dataset, args.aggr_type, args.entity_user_emb_dim, args.relation_emb_dim,
            '-'.join([str(i) for i in eval(args.conv_layer_dims)]), args.lr, args.pretrain)

    args.save_dir = save_dir

    return args