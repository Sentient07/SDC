# argparse.py

def get_argparse_2(parser):
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--config', type=str,  required=True)
    parser.add_argument('--id', type=str, default='1')
    parser.add_argument('--gpus', type=int, default=1, help='input batch size')
    parser.add_argument('--strategy', type=str, default=None, help='input batch size')
    parser.add_argument('--only_test', action='store_true', help='Only test')
    parser.add_argument('--wo_model_save', action='store_true')
    return parser


