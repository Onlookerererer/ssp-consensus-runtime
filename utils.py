import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_training_args():
    parser = argparse.ArgumentParser(description='SEHA')
    parser.add_argument("--dataset", type=str, default="wiki", help="Dataset to use (wiki, nus-wide, INRIA-Websearch, xmedianet)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument('--independent_train_seed', action='store_true',
                        help='Restore the CLI training seed after fixed data preparation')
    parser.add_argument("--partial_length", type=int, default=5) # 2,3,4,5
    parser.add_argument("--MAX_EPOCH", type=int, default=180)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--output_dim", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)   # wiki: 1e-4, nus-wide: 1e-4, INRIA-Websearch: 1e-4, xmedianet: 2e-5
    parser.add_argument("--lamda", type=float, default=0.1) 
    parser.add_argument('--env_weight', type=float, default=0.0,
                        help='Optional constrained environment semantic-gradient penalty')
    parser.add_argument("--ema_decay", type=float, default=0.0) # 0.95
    parser.add_argument('--linear', type=str2bool, default=True)
    parser.add_argument("--GPU", type=int, default=0)
    # Neighbor Refining V1 (all baseline defaults above remain unchanged).
    parser.add_argument('--neighbor_refine', action='store_true')
    parser.add_argument('--neighbor_mode', choices=('off', 'consensus', 'causal_consensus', 'causal_shared', 'causal_invariant', 'similarity_mean', 'causal_reweight', 'disagree', 'agree', 'uniform_q', 'permuted_q', 'permuted_q_left'), default=None)
    parser.add_argument('--neighbor_k', type=int, default=10)
    parser.add_argument('--neighbor_beta', type=float, default=0.2)
    parser.add_argument('--neighbor_margin', type=float, default=0.10)
    parser.add_argument('--neighbor_support_threshold', type=float, default=0.20)
    parser.add_argument('--log_dir', default=None, help='Optional separate experiment log directory')
    parser.add_argument('--reference_mode', choices=('off', 'in_fold', 'out_of_fold'), default='off')
    parser.add_argument('--reference_cache', default=None)
    parser.add_argument('--reference_weight', type=float, default=0.1)
    parser.add_argument('--best_checkpoint', default=None,
                        help='Optional path to save best-validation model+emb and a fixed-input reload probe')
    # Legacy optional experiments below: disabled in the current Consensus command.
    parser.add_argument('--anchor_supervision', action='store_true',
                        help='Add CE on training pairs with singleton image/text candidate intersection')
    parser.add_argument('--anchor_lambda', type=float, default=0.1)
    parser.add_argument('--relation_loss', action='store_true',
                        help='Add candidate-disjoint cross-modal relation loss during training')
    parser.add_argument('--relation_margin', type=float, default=0.2)
    parser.add_argument('--relation_weight', type=float, default=0.1)
    parser.add_argument('--trajectory_log', action='store_true',
                        help='Record detached training states for V5 diagnostics')
    parser.add_argument('--trajectory_dir', default=None)
    parser.add_argument('--equivalence_capture', default=None,
                        help='Optional 2-epoch exact-state smoke output')

    args = parser.parse_args()
    if args.reference_mode != 'off':
        if not args.reference_cache or args.reference_weight != 0.1:
            parser.error('CFSG requires --reference_cache and fixed --reference_weight 0.1')
        if args.neighbor_mode != 'consensus' or not args.independent_train_seed:
            parser.error('CFSG requires --neighbor_mode consensus --independent_train_seed')
        if args.anchor_supervision or args.relation_loss or args.trajectory_log or args.equivalence_capture:
            parser.error('CFSG requires anchor/relation/trajectory/equivalence switches off')
    if args.neighbor_refine and args.neighbor_mode not in (None, 'consensus'):
        parser.error('--neighbor_refine is the legacy alias for --neighbor_mode consensus')
    if args.neighbor_mode is None:
        args.neighbor_mode = 'consensus' if args.neighbor_refine else 'off'
    args.neighbor_refine = args.neighbor_mode != 'off'
    if args.anchor_supervision and args.neighbor_mode != 'consensus':
        parser.error('--anchor_supervision requires --neighbor_mode consensus')
    if args.trajectory_log and (args.neighbor_mode != 'consensus' or not args.trajectory_dir):
        parser.error('--trajectory_log requires consensus and --trajectory_dir')

    return args

# 0.01:{0.544,0.507,0.526}
# 0.1: {0.547,0.509,0.528} 0.2:{0.540,0.509,0.524} 0.3:{0.548,0.512,0.530}
# 1.0: {0.542,0.507,0.524}
# 10.0:{0.532,0.491,0.511}
