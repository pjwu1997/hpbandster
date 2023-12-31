"""
Example 5 - MNIST
=================

Small CNN for MNIST implementet in both Keras and PyTorch.
This example also shows how to log results to disk during the optimization
which is useful for long runs, because intermediate results are directly available
for analysis. It also contains a more realistic search space with different types
of variables to be optimized.

"""
import os
import pickle
import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB, STAT_BOHB, HyperBand, Stat_HyperBand

import logging
logging.basicConfig(level=logging.DEBUG)



parser = argparse.ArgumentParser(description='Example 5 - CNN on MNIST')
parser.add_argument('--min_budget',   type=float, help='Minimum number of epochs for training.',    default=2)
parser.add_argument('--max_budget',   type=float, help='Maximum number of epochs for training.',    default=100)
parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=9)
parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
parser.add_argument('--run_id', type=str, help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.')
parser.add_argument('--nic_name',type=str, help='Which network interface to use for communication.', default='lo')
parser.add_argument('--shared_directory',type=str, help='A directory that is accessible for all processes, e.g. a NFS share.', default='.')
parser.add_argument('--backend',help='Toggles which worker is used. Choose between a pytorch and a keras implementation.', choices=['pytorch', 'keras'], default='pytorch')

args=parser.parse_args()


if args.backend == 'pytorch':
	from example_5_pytorch_worker import PyTorchWorker as worker
else:
	from example_5_keras_worker import KerasWorker as worker

for times in range(5,10):
	# for model in ['HYPERBAND']:
	seed = times
	for model in ['STAT_BOHB', 'BOHB', 'STAT_BAND', 'HYPERBAND']:
		# Every process has to lookup the hostname
		host = hpns.nic_name_to_host(args.nic_name)
		args.shared_directory = './' + model + '/' + str(times) + '/'
		# print(args.shared_directory)
		directory = args.shared_directory
		os.makedirs(os.path.dirname(directory), exist_ok=True)

		if args.worker:
			import time
			time.sleep(5)	# short artificial delay to make sure the nameserver is already running
			w = worker(run_id=args.run_id, host=host, timeout=120)
			
			w.load_nameserver_credentials(working_directory=directory)
			w.run(background=False)
			exit(0)


		# This example shows how to log live results. This is most useful
		# for really long runs, where intermediate results could already be
		# interesting. The core.result submodule contains the functionality to
		# read the two generated files (results.json and configs.json) and
		# create a Result object.
		result_logger = hpres.json_result_logger(directory=directory, overwrite=True)


		# Start a nameserver:
		NS = hpns.NameServer(run_id=args.run_id, host=host, port=0, working_directory=directory)
		ns_host, ns_port = NS.start()

		# Start local worker
		w = worker(run_id=args.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port, timeout=120)
		w.run(background=True)

		# Run an optimizer
		if model == 'BOHB':
			bohb = BOHB(  configspace = worker.get_configspace(seed=seed),
					run_id = args.run_id,
					host=host,
					nameserver=ns_host,
					nameserver_port=ns_port,
					result_logger=result_logger,
					min_budget=args.min_budget, max_budget=args.max_budget, 
				)
			res = bohb.run(n_iterations=args.n_iterations)
			
		elif model == 'STAT_BOHB':
			bohb = STAT_BOHB(  configspace = worker.get_configspace(seed=seed),
						run_id = args.run_id,
						host=host,
						nameserver=ns_host,
						nameserver_port=ns_port,
						result_logger=result_logger,
						min_budget=args.min_budget, max_budget=args.max_budget, 
					)
			res = bohb.run(n_iterations=args.n_iterations)
		
		elif model == 'HYPERBAND':
			bohb = HyperBand(  configspace = worker.get_configspace(seed=seed),
						run_id = args.run_id,
						host=host,
						nameserver=ns_host,
						nameserver_port=ns_port,
						result_logger=result_logger,
						min_budget=args.min_budget, max_budget=args.max_budget, 
					)
			res = bohb.run(n_iterations=args.n_iterations)

		elif model == 'STAT_BAND':
			bohb = Stat_HyperBand(
				        configspace = worker.get_configspace(seed=seed),
						run_id = args.run_id,
						host=host,
						nameserver=ns_host,
						nameserver_port=ns_port,
						result_logger=result_logger,
						min_budget=args.min_budget, max_budget=args.max_budget, 
					)
			res = bohb.run(n_iterations=args.n_iterations)


		# store results
		# directory = args.shared_directory + '/' + model + '/' + str(times) + '/'
		with open(os.path.join(directory, 'results.pkl'), 'wb') as fh:
			pickle.dump(res, fh)

		# shutdown
		bohb.shutdown(shutdown_workers=True)
		NS.shutdown()

