"""
Main training script for NERSC PyTorch examples
"""

# System
import os
import argparse
import logging

# Externals
import yaml
import numpy as np
import torch.distributed as dist

# Locals
from datasets import get_data_loaders
from trainers import get_trainer
from utils.logging import config_logging
from utils.distributed import init_workers

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('train.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/hello.yaml')
    add_arg('-d', '--distributed-backend', choices=['mpi', 'nccl', 'gloo'])
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--gpu', type=int)
    add_arg('--rank-gpu', action='store_true')
    add_arg('--ranks-per-node', type=int, default=8)
    add_arg('--interactive', action='store_true')
    return parser.parse_args()

def load_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def main():
    """Main function"""

    # Initialization
    args = parse_args()
    rank, n_ranks = init_workers(args.distributed_backend)

    # Load configuration
    config = load_config(args.config)
    data_config = config['data_config']
    model_config = config.get('model_config', {})
    train_config = config['train_config']

    # Prepare output directory
    output_dir = config.get('output_dir', None)
    if output_dir is not None:
        output_dir = os.path.expandvars(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    log_file = (os.path.join(output_dir, 'out_%i.log' % rank)
                if output_dir is not None else None)
    config_logging(verbose=args.verbose, log_file=log_file)
    logging.info('Initialized rank %i out of %i', rank, n_ranks)
    if rank == 0:
        logging.info('Configuration: %s' % config)

    # Load the datasets
    is_distributed = args.distributed_backend is not None
    train_data_loader, valid_data_loader = get_data_loaders(
        distributed=is_distributed, **data_config)

    # Load the trainer
    gpu = (rank % args.ranks_per_node) if args.rank_gpu else args.gpu
    if gpu is not None:
        logging.info('Using GPU %i', gpu)
    trainer = get_trainer(name=config['trainer'], distributed=is_distributed,
                          rank=rank, output_dir=output_dir, gpu=gpu)
    # Build the model
    trainer.build_model(**model_config)
    if rank == 0:
        trainer.print_model_summary()

    # Run the training
    summary = trainer.train(train_data_loader=train_data_loader,
                            valid_data_loader=valid_data_loader,
                            **train_config)
    if output_dir is not None:
        trainer.write_summaries()

    # Print some conclusions
    logging.info('Finished training')
    logging.info('Train samples %g time %g s rate %g samples/s',
                 np.mean(summary['train_samples']),
                 np.mean(summary['train_time']),
                 np.mean(summary['train_rate']))
    if valid_data_loader is not None:
        logging.info('Valid samples %g time %g s rate %g samples/s',
                     np.mean(summary['valid_samples']),
                     np.mean(summary['valid_time']),
                     np.mean(summary['valid_rate']))

    # Drop to IPython interactive shell
    if args.interactive and rank==0:
        logging.info('Starting IPython interactive session')
        import IPython
        IPython.embed()

    logging.info('All done!')

if __name__ == '__main__':
    main()
