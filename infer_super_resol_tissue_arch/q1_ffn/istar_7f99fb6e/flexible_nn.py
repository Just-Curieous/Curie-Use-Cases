import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim import Adam


class FlexibleFeedForward(nn.Module):
    """
    A flexible feedforward layer implementation that supports:
    - Customizable activation functions
    - Optional residual connections
    - Optional dropout
    - Optional batch normalization
    """
    def __init__(
            self, n_inp, n_out,
            activation=None, residual=False,
            dropout_rate=0.0, batch_norm=False):
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
                # For 3D tensors (batch_size, seq_len, features)
                batch_size, seq_len, features = y.shape
                y = y.reshape(-1, features)
                y = self.batch_norm(y)
                y = y.reshape(batch_size, seq_len, features)
            else:
                # For 2D tensors (batch_size, features)
                y = self.batch_norm(y)
        
        # Apply activation
        y = self.activation(y)
        
        # Apply dropout if enabled
        if self.use_dropout:
            y = self.dropout(y)
        
        # Apply residual connection if enabled and dimensions match
        if self.residual and x.shape == y.shape:
            y = y + x
            
        return y


class ELU(nn.Module):
    """
    Custom ELU activation with alpha and beta parameters
    """
    def __init__(self, alpha, beta):
        super().__init__()
        self.activation = nn.ELU(alpha=alpha, inplace=True)
        self.beta = beta

    def forward(self, x):
        return self.activation(x) + self.beta


class FlexibleNeuralNetwork(pl.LightningModule):
    """
    A flexible neural network model class that supports:
    - Configurable hidden layer architecture (number of layers and units per layer)
    - Choice of activation functions
    - Various optimizer configurations including Adam with beta parameters
    - Adjustable learning rate
    - Batch size configuration
    - Weight decay regularization
    """
    def __init__(
            self, n_inp, n_out, 
            hidden_layers="256,256,256,256",
            activation="leaky_relu", activation_params={"negative_slope": 0.1},
            output_activation="elu", output_activation_params={"alpha": 0.01, "beta": 0.01},
            residual=False, dropout_rate=0.0, batch_norm=False,
            learning_rate=1e-4, weight_decay=0.0,
            optimizer="adam", optimizer_params={"betas": (0.9, 0.999), "eps": 1e-8}):
        super().__init__()
        
        self.n_inp = n_inp
        self.n_out = n_out
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        
        # Parse hidden layers configuration
        if isinstance(hidden_layers, str):
            hidden_layers = [int(x) for x in hidden_layers.split(",")]
        
        # Create activation function based on name
        if activation == "leaky_relu":
            act_fn = nn.LeakyReLU(negative_slope=activation_params.get("negative_slope", 0.1), inplace=True)
        elif activation == "relu":
            act_fn = nn.ReLU(inplace=True)
        elif activation == "tanh":
            act_fn = nn.Tanh()
        elif activation == "sigmoid":
            act_fn = nn.Sigmoid()
        elif activation == "elu":
            act_fn = nn.ELU(alpha=activation_params.get("alpha", 1.0), inplace=True)
        else:
            act_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        # Create output activation function
        if output_activation == "elu":
            out_act_fn = ELU(
                alpha=output_activation_params.get("alpha", 0.01),
                beta=output_activation_params.get("beta", 0.01)
            )
        elif output_activation == "linear":
            out_act_fn = nn.Identity()
        elif output_activation == "sigmoid":
            out_act_fn = nn.Sigmoid()
        elif output_activation == "tanh":
            out_act_fn = nn.Tanh()
        else:
            out_act_fn = ELU(alpha=0.01, beta=0.01)
        
        # Build the network architecture
        layers = []
        prev_size = n_inp
        
        for layer_size in hidden_layers:
            layers.append(
                FlexibleFeedForward(
                    prev_size, layer_size,
                    activation=act_fn,
                    residual=residual and prev_size == layer_size,
                    dropout_rate=dropout_rate,
                    batch_norm=batch_norm
                )
            )
            prev_size = layer_size
        
        self.net_lat = nn.Sequential(*layers)
        
        # Output layer
        self.net_out = FlexibleFeedForward(
            prev_size, n_out,
            activation=out_act_fn,
            residual=False,
            dropout_rate=0.0,
            batch_norm=False
        )
        
        self.save_hyperparameters()
        
        # Initialize RMSE log file with detailed configuration
        with open('rmse_results.txt', 'w') as f:
            f.write('# Model Configuration\n')
            f.write(f'# Input dimension: {n_inp}\n')
            f.write(f'# Output dimension: {n_out}\n')
            f.write(f'# Hidden layers: {hidden_layers}\n')
            f.write(f'# Activation: {activation} {activation_params}\n')
            f.write(f'# Output activation: {output_activation} {output_activation_params}\n')
            f.write(f'# Residual connections: {residual}\n')
            f.write(f'# Dropout rate: {dropout_rate}\n')
            f.write(f'# Batch normalization: {batch_norm}\n')
            f.write(f'# Learning rate: {learning_rate}\n')
            f.write(f'# Weight decay: {weight_decay}\n')
            f.write(f'# Optimizer: {optimizer} {optimizer_params}\n')
            f.write('# Training Metrics\n')
            f.write('epoch,rmse,loss,learning_rate,batch_size\n')

    def inp_to_lat(self, x):
        return self.net_lat.forward(x)

    def lat_to_out(self, x, indices=None):
        x = self.net_out.forward(x, indices)
        return x

    def forward(self, x, indices=None):
        x = self.inp_to_lat(x)
        x = self.lat_to_out(x, indices)
        return x

    def training_step(self, batch, batch_idx):
        x, y_mean = batch
        y_pred = self.forward(x)
        y_mean_pred = y_pred.mean(-2)
        mse = ((y_mean_pred - y_mean)**2).mean()
        loss = mse
        rmse = mse**0.5
        self.log('rmse', rmse, prog_bar=True)
        
        # Get current learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        batch_size = x.shape[0]
        
        # Log detailed metrics to file
        if self.current_epoch % 10 == 0 and batch_idx == 0: 
            with open('rmse_results.txt', 'a') as f:
                f.write(f'{self.current_epoch},{rmse:.6f},{loss:.6f},{current_lr:.6f},{batch_size}\n')
            
        return loss

    def configure_optimizers(self):
        if self.optimizer.lower() == "adam":
            optimizer = Adam(
                self.parameters(), 
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=self.optimizer_params.get("betas", (0.9, 0.999)),
                eps=self.optimizer_params.get("eps", 1e-8)
            )
        else:
            # Default to Adam if optimizer not recognized
            optimizer = Adam(
                self.parameters(), 
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        return optimizer