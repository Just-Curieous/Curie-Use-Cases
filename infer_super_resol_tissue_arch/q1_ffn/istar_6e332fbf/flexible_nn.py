import argparse
import multiprocessing
import time
from typing import List, Optional, Union, Dict, Any

import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch import nn
import numpy as np

from impute_by_basic import get_gene_counts, get_embeddings, get_locs
from utils import read_lines, read_string, save_pickle
from image import get_disk_mask
from train import get_model as train_load_model
from visual import plot_matrix, plot_spot_masked_image


class FeedForward(nn.Module):
    """Flexible feedforward layer with various activation functions and optional residual connections."""
    
    def __init__(
            self, n_inp: int, n_out: int,
            activation: Optional[nn.Module] = None, 
            residual: bool = False,
            dropout_rate: float = 0.0,
            batch_norm: bool = False):
        super().__init__()
        self.linear = nn.Linear(n_inp, n_out)
        
        # Default activation if none provided
        if activation is None:
            activation = nn.LeakyReLU(0.1, inplace=True)
        self.activation = activation
        
        # Optional components
        self.residual = residual
        self.use_dropout = dropout_rate > 0
        self.use_batch_norm = batch_norm
        
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)
        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(n_out)

    def forward(self, x, indices=None):
        if indices is None:
            y = self.linear(x)
        else:
            weight = self.linear.weight[indices]
            bias = self.linear.bias[indices]
            y = nn.functional.linear(x, weight, bias)
        
        # Apply batch normalization if enabled
        if self.use_batch_norm:
            # Handle different input shapes
            if len(y.shape) == 3:
                # Reshape for BatchNorm1d
                orig_shape = y.shape
                y = y.reshape(-1, y.shape[-1])
                y = self.batch_norm(y)
                y = y.reshape(orig_shape)
            else:
                y = self.batch_norm(y)
        
        # Apply activation
        y = self.activation(y)
        
        # Apply dropout if enabled
        if self.use_dropout:
            y = self.dropout(y)
        
        # Add residual connection if enabled and dimensions match
        if self.residual and x.shape[-1] == y.shape[-1]:
            y = y + x
            
        return y


class ELU(nn.Module):
    """Custom ELU activation with beta parameter."""
    
    def __init__(self, alpha: float, beta: float):
        super().__init__()
        self.activation = nn.ELU(alpha=alpha, inplace=True)
        self.beta = beta

    def forward(self, x):
        return self.activation(x) + self.beta


class FlexibleNeuralNetwork(pl.LightningModule):
    """Flexible neural network architecture for imputation tasks."""
    
    def __init__(
            self, 
            n_inp: int, 
            n_out: int,
            hidden_layers: List[int] = [256, 256, 256, 256],
            activation: str = 'leaky_relu',
            activation_params: Dict[str, float] = {'negative_slope': 0.1},
            output_activation: str = 'elu',
            output_activation_params: Dict[str, float] = {'alpha': 0.01, 'beta': 0.01},
            optimizer: str = 'adam',
            optimizer_params: Dict[str, Any] = {'lr': 1e-4},
            dropout_rate: float = 0.0,
            batch_norm: bool = False,
            residual_connections: bool = False,
            weight_decay: float = 0.0):
        super().__init__()
        
        self.save_hyperparameters()
        self.n_inp = n_inp
        self.n_out = n_out
        
        # Configure activation function
        self.activation_name = activation
        self.activation_params = activation_params
        self.output_activation_name = output_activation
        self.output_activation_params = output_activation_params
        
        # Configure optimizer
        self.optimizer_name = optimizer
        self.optimizer_params = optimizer_params
        
        # Configure regularization
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.residual_connections = residual_connections
        self.weight_decay = weight_decay
        
        # Build network architecture
        self._build_network(hidden_layers)
        
        # Initialize results file
        self._init_results_file()
        
    def _get_activation(self, name: str, params: Dict[str, float]) -> nn.Module:
        """Create activation function based on name and parameters."""
        if name == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=params.get('negative_slope', 0.1), inplace=True)
        elif name == 'relu':
            return nn.ReLU(inplace=True)
        elif name == 'elu':
            if 'beta' in params:
                return ELU(alpha=params.get('alpha', 1.0), beta=params.get('beta', 0.0))
            else:
                return nn.ELU(alpha=params.get('alpha', 1.0), inplace=True)
        elif name == 'gelu':
            return nn.GELU()
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {name}")
    
    def _build_network(self, hidden_layers: List[int]) -> None:
        """Build the neural network architecture."""
        # Create hidden layers
        layers = []
        input_dim = self.n_inp
        
        for i, hidden_dim in enumerate(hidden_layers):
            activation = self._get_activation(self.activation_name, self.activation_params)
            use_residual = self.residual_connections and input_dim == hidden_dim
            
            layers.append(
                FeedForward(
                    n_inp=input_dim, 
                    n_out=hidden_dim,
                    activation=activation,
                    residual=use_residual,
                    dropout_rate=self.dropout_rate,
                    batch_norm=self.batch_norm
                )
            )
            input_dim = hidden_dim
        
        self.net_lat = nn.Sequential(*layers)
        
        # Create output layer
        output_activation = self._get_activation(
            self.output_activation_name, 
            self.output_activation_params
        )
        
        self.net_out = FeedForward(
            n_inp=input_dim,
            n_out=self.n_out,
            activation=output_activation,
            residual=False,
            dropout_rate=0.0,  # No dropout in output layer
            batch_norm=False   # No batch norm in output layer
        )
    
    def _init_results_file(self) -> None:
        """Initialize the results file with model configuration."""
        with open('rmse_results.txt', 'w') as f:
            f.write('# Model Configuration\n')
            f.write(f'# Input dimension: {self.n_inp}\n')
            f.write(f'# Output dimension: {self.n_out}\n')
            f.write(f'# Hidden layers: {self.hparams.hidden_layers}\n')
            f.write(f'# Activation: {self.activation_name} {self.activation_params}\n')
            f.write(f'# Output activation: {self.output_activation_name} {self.output_activation_params}\n')
            f.write(f'# Optimizer: {self.optimizer_name} {self.optimizer_params}\n')
            f.write(f'# Dropout rate: {self.dropout_rate}\n')
            f.write(f'# Batch normalization: {self.batch_norm}\n')
            f.write(f'# Residual connections: {self.residual_connections}\n')
            f.write(f'# Weight decay: {self.weight_decay}\n')
            f.write('# Training Metrics\n')
            f.write('epoch,rmse,loss,learning_rate,batch_size,time\n')
    
    def inp_to_lat(self, x):
        """Forward pass through the latent network."""
        return self.net_lat.forward(x)

    def lat_to_out(self, x, indices=None):
        """Forward pass through the output network."""
        x = self.net_out.forward(x, indices)
        return x

    def forward(self, x, indices=None):
        """Full forward pass through the network."""
        x = self.inp_to_lat(x)
        x = self.lat_to_out(x, indices)
        return x

    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y_mean = batch
        y_pred = self.forward(x)
        y_mean_pred = y_pred.mean(-2)
        
        # Calculate loss
        mse = ((y_mean_pred - y_mean)**2).mean()
        loss = mse
        rmse = mse**0.5
        
        # Log metrics
        self.log('rmse', rmse, prog_bar=True)
        
        # Get current learning rate and batch size
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        batch_size = x.shape[0]
        
        # Log detailed metrics to file every 10 epochs
        if self.current_epoch % 10 == 0 and batch_idx == 0:
            with open('rmse_results.txt', 'a') as f:
                f.write(f'{self.current_epoch},{rmse:.6f},{loss:.6f},{current_lr:.6f},{batch_size},{time.time():.1f}\n')
        
        return loss

    def configure_optimizers(self):
        """Configure optimizer based on parameters."""
        if self.optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(), 
                **self.optimizer_params,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                **self.optimizer_params,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(), 
                **self.optimizer_params,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        
        return optimizer


class SpotDataset(Dataset):
    """Dataset for spot data."""

    def __init__(self, x_all, y, locs, radius):
        super().__init__()
        mask = get_disk_mask(radius)
        x = get_patches_flat(x_all, locs, mask)
        isin = np.isfinite(x).all((-1, -2))
        self.x = x[isin]
        self.y = y[isin]
        self.locs = locs[isin]
        self.size = x_all.shape[:2]
        self.radius = radius
        self.mask = mask

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def show(self, channel_x, channel_y, prefix):
        mask = self.mask
        size = self.size
        locs = self.locs
        xs = self.x
        ys = self.y

        plot_spot_masked_image(
                locs=locs, values=xs[:, :, channel_x], mask=mask, size=size,
                outfile=f'{prefix}x{channel_x:04d}.png')

        plot_spot_masked_image(
                locs=locs, values=ys[:, channel_y], mask=mask, size=size,
                outfile=f'{prefix}y{channel_y:04d}.png')


def get_disk(img, ij, radius):
    i, j = ij
    patch = img[i-radius:i+radius, j-radius:j+radius]
    disk_mask = get_disk_mask(radius)
    patch[~disk_mask] = 0.0
    return patch


def get_patches_flat(img, locs, mask):
    shape = np.array(mask.shape)
    center = shape // 2
    r = np.stack([-center, shape-center], -1)  # offset
    x_list = []
    for s in locs:
        patch = img[
                s[0]+r[0][0]:s[0]+r[0][1],
                s[1]+r[1][0]:s[1]+r[1][1]]
        if mask.all():
            x = patch
        else:
            x = patch[mask]
        x_list.append(x)
    x_list = np.stack(x_list)
    return x_list


def add_coords(embs):
    coords = np.stack(np.meshgrid(
            np.linspace(-1, 1, embs.shape[0]),
            np.linspace(-1, 1, embs.shape[1]),
            indexing='ij'), -1)
    coords = coords.astype(embs.dtype)
    mask = np.isfinite(embs).all(-1)
    coords[~mask] = np.nan
    embs = np.concatenate([embs, coords], -1)
    return embs


def get_data(prefix):
    gene_names = read_lines(f'{prefix}gene-names.txt')
    cnts = get_gene_counts(prefix)
    cnts = cnts[gene_names]
    embs = get_embeddings(prefix)
    locs = get_locs(prefix, target_shape=embs.shape[:2])
    return embs, cnts, locs


def get_model_kwargs(kwargs):
    return get_model(**kwargs)


def get_model(
        x, y, locs, radius, prefix, batch_size, epochs, 
        model_config=None, load_saved=False, device='cuda'):
    """Get or train a model."""
    print('x:', x.shape, ', y:', y.shape)

    x = x.copy()

    dataset = SpotDataset(x, y, locs, radius)
    dataset.show(
            channel_x=0, channel_y=0,
            prefix=f'{prefix}training-data-plots/')
    
    # Default model configuration for control group
    if model_config is None:
        model_config = {
            'hidden_layers': [256, 256, 256, 256],
            'activation': 'leaky_relu',
            'activation_params': {'negative_slope': 0.1},
            'output_activation': 'elu',
            'output_activation_params': {'alpha': 0.01, 'beta': 0.01},
            'optimizer': 'adam',
            'optimizer_params': {'lr': 1e-4},
            'dropout_rate': 0.0,
            'batch_norm': False,
            'residual_connections': False,
            'weight_decay': 0.0
        }
    
    model_kwargs = {
        'n_inp': x.shape[-1],
        'n_out': y.shape[-1],
        **model_config
    }
    
    model = train_load_model(
            model_class=FlexibleNeuralNetwork,
            model_kwargs=model_kwargs,
            dataset=dataset, prefix=prefix,
            batch_size=batch_size, epochs=epochs,
            load_saved=load_saved, device=device)
    
    model.eval()
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    return model, dataset


def normalize(embs, cnts):
    """Normalize embeddings and counts."""
    embs = embs.copy()
    cnts = cnts.copy()

    embs_mean = np.nanmean(embs, (0, 1))
    embs_std = np.nanstd(embs, (0, 1))
    embs -= embs_mean
    embs /= embs_std + 1e-12

    cnts_min = cnts.min(0)
    cnts_max = cnts.max(0)
    cnts -= cnts_min
    cnts /= (cnts_max - cnts_min) + 1e-12

    return embs, cnts, (embs_mean, embs_std), (cnts_min, cnts_max)


def show_results(x, names, prefix):
    """Show results for specific genes."""
    for name in ['CD19', 'MS4A1', 'ERBB2', 'GNAS']:
        if name in names:
            idx = np.where(names == name)[0][0]
            plot_matrix(x[..., idx], prefix+name+'.png')


def predict_single_out(model, z, indices, names, y_range):
    """Predict output for a single latent representation."""
    z = torch.tensor(z, device=model.device)
    y = model.lat_to_out(z, indices=indices)
    y = y.cpu().detach().numpy()
    y *= y_range[:, 1] - y_range[:, 0]
    y += y_range[:, 0]
    return y


def predict_single_lat(model, x):
    """Predict latent representation for a single input."""
    x = torch.tensor(x, device=model.device)
    z = model.inp_to_lat(x)
    z = z.cpu().detach().numpy()
    return z


def predict(
        model_states, x_batches, name_list, y_range, prefix,
        device='cuda'):
    """Predict outputs for all inputs."""
    batch_size_outcome = 100

    model_states = [mod.to(device) for mod in model_states]

    # Get features of second last layer
    z_states_batches = [
            [predict_single_lat(mod, x_bat) for mod in model_states]
            for x_bat in x_batches]
    z_point = np.concatenate([
        np.median(z_states, 0)
        for z_states in z_states_batches])
    z_dict = dict(cls=z_point.transpose(2, 0, 1))
    save_pickle(
            z_dict,
            prefix+'embeddings-gene.pickle')
    del z_point

    # Predict and save y by batches in outcome dimension
    idx_list = np.arange(len(name_list))
    n_groups_outcome = len(idx_list) // batch_size_outcome + 1
    idx_groups = np.array_split(idx_list, n_groups_outcome)
    for idx_grp in idx_groups:
        name_grp = name_list[idx_grp]
        y_ran = y_range[idx_grp]
        y_grp = np.concatenate([
            np.median([
                predict_single_out(mod, z, idx_grp, name_grp, y_ran)
                for mod, z in zip(model_states, z_states)], 0)
            for z_states in z_states_batches])
        for i, name in enumerate(name_grp):
            save_pickle(y_grp[..., i], f'{prefix}cnts-super/{name}.pickle')


def impute(
        embs, cnts, locs, radius, epochs, batch_size, prefix,
        model_config=None, n_states=1, load_saved=False, 
        device='cuda', n_jobs=1):
    """Impute gene expression values."""
    names = cnts.columns
    cnts = cnts.to_numpy()
    cnts = cnts.astype(np.float32)

    __, cnts, __, (cnts_min, cnts_max) = normalize(embs, cnts)

    kwargs_list = [
            dict(
                x=embs, y=cnts, locs=locs, radius=radius,
                batch_size=batch_size, epochs=epochs,
                model_config=model_config,
                prefix=f'{prefix}states/{i:02d}/',
                load_saved=load_saved, device=device)
            for i in range(n_states)]

    if n_jobs is None or n_jobs < 1:
        n_jobs = n_states
    if n_jobs == 1:
        out_list = [get_model_kwargs(kwargs) for kwargs in kwargs_list]
    else:
        with multiprocessing.Pool(processes=n_jobs) as pool:
            out_list = pool.map(get_model_kwargs, kwargs_list)

    model_list = [out[0] for out in out_list]
    dataset_list = [out[1] for out in out_list]
    mask_size = dataset_list[0].mask.sum()

    cnts_range = np.stack([cnts_min, cnts_max], -1)
    cnts_range /= mask_size

    batch_size_row = 50
    n_batches_row = embs.shape[0] // batch_size_row + 1
    embs_batches = np.array_split(embs, n_batches_row)
    del embs
    predict(
            model_states=model_list, x_batches=embs_batches,
            name_list=names, y_range=cnts_range,
            prefix=prefix, device=device)


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Flexible Neural Network for Imputation')
    
    # Required arguments
    parser.add_argument('prefix', type=str, help='Data prefix')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=400, help='Number of training epochs')
    parser.add_argument('--n-states', type=int, default=1, help='Number of model states')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--n-jobs', type=int, default=1, help='Number of parallel jobs')
    parser.add_argument('--load-saved', action='store_true', help='Load saved model if available')
    
    # Model architecture
    parser.add_argument('--hidden-layers', type=str, default='256,256,256,256', 
                        help='Hidden layer dimensions (comma-separated)')
    parser.add_argument('--activation', type=str, default='leaky_relu', 
                        help='Activation function (leaky_relu, relu, elu, gelu, tanh, sigmoid)')
    parser.add_argument('--leaky-slope', type=float, default=0.1, 
                        help='Negative slope for LeakyReLU')
    parser.add_argument('--output-activation', type=str, default='elu', 
                        help='Output activation function')
    parser.add_argument('--elu-alpha', type=float, default=0.01, 
                        help='Alpha parameter for ELU')
    parser.add_argument('--elu-beta', type=float, default=0.01, 
                        help='Beta parameter for ELU')
    
    # Optimizer
    parser.add_argument('--optimizer', type=str, default='adam', 
                        help='Optimizer (adam, adamw, sgd)')
    parser.add_argument('--learning-rate', type=float, default=1e-4, 
                        help='Learning rate')
    
    # Regularization
    parser.add_argument('--dropout', type=float, default=0.0, 
                        help='Dropout rate')
    parser.add_argument('--batch-norm', action='store_true', 
                        help='Use batch normalization')
    parser.add_argument('--residual', action='store_true', 
                        help='Use residual connections')
    parser.add_argument('--weight-decay', type=float, default=0.0, 
                        help='Weight decay for regularization')
    
    args = parser.parse_args()
    return args


def main():
    """Main function."""
    args = get_args()
    
    # Get data
    embs, cnts, locs = get_data(args.prefix)
    
    # Process radius
    factor = 16
    radius = int(read_string(f'{args.prefix}radius.txt'))
    radius = radius / factor
    
    # Determine batch size
    n_train = cnts.shape[0]
    batch_size = min(128, n_train//16)
    
    # Parse hidden layers
    hidden_layers = [int(x) for x in args.hidden_layers.split(',')]
    
    # Create model configuration
    model_config = {
        'hidden_layers': hidden_layers,
        'activation': args.activation,
        'activation_params': {'negative_slope': args.leaky_slope} if args.activation == 'leaky_relu' else {},
        'output_activation': args.output_activation,
        'output_activation_params': {'alpha': args.elu_alpha, 'beta': args.elu_beta} if args.output_activation == 'elu' else {},
        'optimizer': args.optimizer,
        'optimizer_params': {'lr': args.learning_rate},
        'dropout_rate': args.dropout,
        'batch_norm': args.batch_norm,
        'residual_connections': args.residual,
        'weight_decay': args.weight_decay
    }
    
    # For control group, use baseline configuration
    if args.hidden_layers == '256,256,256,256' and args.activation == 'leaky_relu' and args.leaky_slope == 0.1 and \
       args.output_activation == 'elu' and args.elu_alpha == 0.01 and args.elu_beta == 0.01 and \
       args.optimizer == 'adam' and args.learning_rate == 1e-4 and \
       args.dropout == 0.0 and not args.batch_norm and not args.residual and args.weight_decay == 0.0:
        print("Using baseline configuration for control group")
    
    # Run imputation
    start_time = time.time()
    impute(
        embs=embs, cnts=cnts, locs=locs, radius=radius,
        epochs=args.epochs, batch_size=batch_size,
        model_config=model_config,
        n_states=args.n_states, prefix=args.prefix,
        load_saved=args.load_saved,
        device=args.device, n_jobs=args.n_jobs)
    end_time = time.time()
    
    # Log total execution time
    with open('rmse_results.txt', 'a') as f:
        f.write(f'\n# Total execution time: {end_time - start_time:.2f} seconds\n')


if __name__ == '__main__':
    main()