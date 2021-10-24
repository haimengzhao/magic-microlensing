import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt

import time
import argparse
import numpy as np
from random import SystemRandom

import torch
import torch.nn as nn
import torch.optim as optim

import model.utils as utils
from model.latent_ode import create_LatentODE_model

from model.preprocessing import parse_datasets
from model.neural_ode import ODEFunc
from model.neural_ode import DiffeqSolver

from model.utils import compute_loss_all_batches

parser = argparse.ArgumentParser('Latent ODE')
parser.add_argument('--niters', type=int, default=300)
parser.add_argument('--lr',  type=float, default=1e-2, help="Starting learning rate")
parser.add_argument('-b', '--batch-size', type=int, default=50)
parser.add_argument('--viz', action='store_true', help="Show plots while training")

parser.add_argument('--dataset', type=str, default='/work/hmzhao/irregular-lc/random-even-batch-0.h5', help="Path for dataset")
parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('-r', '--random-seed', type=int, default=414, help="Random_seed")

parser.add_argument('-l', '--latents', type=int, default=16, help="Dim of the latent state")
parser.add_argument('--rec-layers', type=int, default=1, help="Number of layers in ODE func in recognition ODE")
parser.add_argument('--gen-layers', type=int, default=1, help="Number of layers in ODE func in generative ODE")

parser.add_argument('-u', '--units', type=int, default=100, help="Number of units per layer in ODE func")
parser.add_argument('-g', '--gru-units', type=int, default=100, help="Number of units per layer in each of GRU update networks")

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
file_name = os.path.basename(__file__)[:-3]
utils.makedirs(args.save)

#####################################################################################################

if __name__ == '__main__':
	torch.manual_seed(args.random_seed)
	np.random.seed(args.random_seed)

	experimentID = args.load
	if experimentID is None:
		# Make a new experiment ID
		experimentID = int(SystemRandom().random()*100000)
	ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + '.ckpt')

	start = time.time()
	print("Sampling dataset of {} training examples".format(args.n))
	
	input_command = sys.argv
	ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
	if len(ind) == 1:
		ind = ind[0]
		input_command = input_command[:ind] + input_command[(ind+2):]
	input_command = " ".join(input_command)

	utils.makedirs("results/")

	##################################################################
	data_obj = parse_datasets(args, device)
	input_dim = data_obj["input_dim"]

	classif_per_tp = False
	if ("classif_per_tp" in data_obj):
		# do classification per time point rather than on a time series as a whole
		classif_per_tp = data_obj["classif_per_tp"]

	if args.classif and (args.dataset == "hopper" or args.dataset == "periodic"):
		raise Exception("Classification task is not available for MuJoCo and 1d datasets")

	n_labels = 1
	if args.classif:
		if ("n_labels" in data_obj):
			n_labels = data_obj["n_labels"]
		else:
			raise Exception("Please provide number of labels for classification task")

	##################################################################
	# Create the model
	obsrv_std = 0.01
	if args.dataset == "hopper":
		obsrv_std = 1e-3 

	obsrv_std = torch.Tensor([obsrv_std]).to(device)

	z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))

	if args.rnn_vae:
		if args.poisson:
			print("Poisson process likelihood not implemented for RNN-VAE: ignoring --poisson")

		# Create RNN-VAE model
		model = RNN_VAE(input_dim, args.latents, 
			device = device, 
			rec_dims = args.rec_dims, 
			concat_mask = True, 
			obsrv_std = obsrv_std,
			z0_prior = z0_prior,
			use_binary_classif = args.classif,
			classif_per_tp = classif_per_tp,
			linear_classifier = args.linear_classif,
			n_units = args.units,
			input_space_decay = args.input_decay,
			cell = args.rnn_cell,
			n_labels = n_labels,
			train_classif_w_reconstr = (args.dataset == "physionet")
			).to(device)


	elif args.classic_rnn:
		if args.poisson:
			print("Poisson process likelihood not implemented for RNN: ignoring --poisson")

		if args.extrap:
			raise Exception("Extrapolation for standard RNN not implemented")
		# Create RNN model
		model = Classic_RNN(input_dim, args.latents, device, 
			concat_mask = True, obsrv_std = obsrv_std,
			n_units = args.units,
			use_binary_classif = args.classif,
			classif_per_tp = classif_per_tp,
			linear_classifier = args.linear_classif,
			input_space_decay = args.input_decay,
			cell = args.rnn_cell,
			n_labels = n_labels,
			train_classif_w_reconstr = (args.dataset == "physionet")
			).to(device)
	elif args.ode_rnn:
		# Create ODE-GRU model
		n_ode_gru_dims = args.latents
				
		if args.poisson:
			print("Poisson process likelihood not implemented for ODE-RNN: ignoring --poisson")

		if args.extrap:
			raise Exception("Extrapolation for ODE-RNN not implemented")

		ode_func_net = utils.create_net(n_ode_gru_dims, n_ode_gru_dims, 
			n_layers = args.rec_layers, n_units = args.units, nonlinear = nn.Tanh)

		rec_ode_func = ODEFunc(
			input_dim = input_dim, 
			latent_dim = n_ode_gru_dims,
			ode_func_net = ode_func_net,
			device = device).to(device)

		z0_diffeq_solver = DiffeqSolver(input_dim, rec_ode_func, "euler", args.latents, 
			odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)
	
		model = ODE_RNN(input_dim, n_ode_gru_dims, device = device, 
			z0_diffeq_solver = z0_diffeq_solver, n_gru_units = args.gru_units,
			concat_mask = True, obsrv_std = obsrv_std,
			use_binary_classif = args.classif,
			classif_per_tp = classif_per_tp,
			n_labels = n_labels,
			train_classif_w_reconstr = (args.dataset == "physionet")
			).to(device)
	elif args.latent_ode:
		model = create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device, 
			classif_per_tp = classif_per_tp,
			n_labels = n_labels)
	else:
		raise Exception("Model not specified")

	##################################################################

	if args.viz:
		viz = Visualizations(device)

	##################################################################
	
	#Load checkpoint and evaluate the model
	if args.load is not None:
		utils.get_ckpt_model(ckpt_path, model, device)
		exit()

	##################################################################
	# Training

	log_path = "logs/" + file_name + "_" + str(experimentID) + ".log"
	if not os.path.exists("logs/"):
		utils.makedirs("logs/")
	logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
	logger.info(input_command)

	optimizer = optim.Adamax(model.parameters(), lr=args.lr)

	num_batches = data_obj["n_train_batches"]

	for itr in range(1, num_batches * (args.niters + 1)):
		optimizer.zero_grad()
		utils.update_learning_rate(optimizer, decay_rate = 0.999, lowest = args.lr / 10)

		wait_until_kl_inc = 10
		if itr // num_batches < wait_until_kl_inc:
			kl_coef = 0.
		else:
			kl_coef = (1-0.99** (itr // num_batches - wait_until_kl_inc))

		batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
		train_res = model.compute_all_losses(batch_dict, n_traj_samples = 3, kl_coef = kl_coef)
		train_res["loss"].backward()
		optimizer.step()

		n_iters_to_viz = 1
		if itr % (n_iters_to_viz * num_batches) == 0:
			with torch.no_grad():

				test_res = compute_loss_all_batches(model, 
					data_obj["test_dataloader"], args,
					n_batches = data_obj["n_test_batches"],
					experimentID = experimentID,
					device = device,
					n_traj_samples = 3, kl_coef = kl_coef)

				message = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Loss {:.6f} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
					itr//num_batches, 
					test_res["loss"].detach(), test_res["likelihood"].detach(), 
					test_res["kl_first_p"], test_res["std_first_p"])
		 	
				logger.info("Experiment " + str(experimentID))
				logger.info(message)
				logger.info("KL coef: {}".format(kl_coef))
				logger.info("Train loss (one batch): {}".format(train_res["loss"].detach()))
				logger.info("Train CE loss (one batch): {}".format(train_res["ce_loss"].detach()))
				
				if "auc" in test_res:
					logger.info("Classification AUC (TEST): {:.4f}".format(test_res["auc"]))

				if "mse" in test_res:
					logger.info("Test MSE: {:.4f}".format(test_res["mse"]))

				if "accuracy" in train_res:
					logger.info("Classification accuracy (TRAIN): {:.4f}".format(train_res["accuracy"]))

				if "accuracy" in test_res:
					logger.info("Classification accuracy (TEST): {:.4f}".format(test_res["accuracy"]))

				if "pois_likelihood" in test_res:
					logger.info("Poisson likelihood: {}".format(test_res["pois_likelihood"]))

				if "ce_loss" in test_res:
					logger.info("CE loss: {}".format(test_res["ce_loss"]))

			torch.save({
				'args': args,
				'state_dict': model.state_dict(),
			}, ckpt_path)


			# Plotting
			if args.viz:
				with torch.no_grad():
					test_dict = utils.get_next_batch(data_obj["test_dataloader"])

					print("plotting....")
					if isinstance(model, LatentODE) and (args.dataset == "periodic"): #and not args.classic_rnn and not args.ode_rnn:
						plot_id = itr // num_batches // n_iters_to_viz
						viz.draw_all_plots_one_dim(test_dict, model, 
							plot_name = file_name + "_" + str(experimentID) + "_{:03d}".format(plot_id) + ".png",
						 	experimentID = experimentID, save=True)
						plt.pause(0.01)
	torch.save({
		'args': args,
		'state_dict': model.state_dict(),
	}, ckpt_path)

