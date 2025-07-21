# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import concurrent.futures

import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
import pickle
import json
import onnxruntime as ort

TensorBatch = List[torch.Tensor]



current_dir = os.path.split(os.path.abspath(__file__))[0]

print("current_dir: ", current_dir)

project_root_path = current_dir.rsplit('/', 1)[0]

print("project_root_path: ", project_root_path)

#pickle_path = os.path.join(project_root_path, 'training_dataset_pickle/v8.pickle')
pickle_path = os.path.join(project_root_path, 'training_dataset_pickle/v18_rand_20Per.pickle')
print("pickle_path: ", pickle_path)

evaluation_dataset_path = os.path.join(project_root_path, 'ALLdatasets', 'evaluate')

print("evaluation_dataset_path: ", evaluation_dataset_path)





ENUM = 600  # every 5 evaluation set
small_evaluation_datasets = []
policy_dir_names = os.listdir(evaluation_dataset_path)
for p_t in policy_dir_names:
    policy_type_dir = os.path.join(evaluation_dataset_path, p_t)
    for e_f_name in os.listdir(policy_type_dir)[:ENUM]:
        e_f_path = os.path.join(policy_type_dir, e_f_name)
        small_evaluation_datasets.append(e_f_path)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
USE_WANDB = 1
b_in_Mb = 1e6

MAX_ACTION = 20  # Mbps
STATE_DIM = 150
ACTION_DIM = 1

EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "v14"
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(2e3)  # How often (time steps) we evaluate
    #max_timesteps: int = int(5e4)
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = './checkpoints_iql'  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    # IQL
    buffer_size: int = 60_000_000  # Replay buffer size
    #batch_size: int = 512  # Batch size for all networks
    batch_size: int = 2048  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    #tau: float = 0.004
    beta: float = 3.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    #beta: float = 4.0
    iql_tau: float = 0.7  # Coefficient for asymmetric loss
    iql_deterministic: bool = False  # Use deterministic actor
    vf_lr: float = 2e-4  # V function learning rate
    qf_lr: float = 2e-4  # Critic learning rate
    actor_lr: float = 2e-4  # Actor learning rate
    #actor_lr: float = 2e-4
    actor_dropout: Optional[float] = None  # Adroit uses dropout for policy network
    # Wandb logging
    project: str = "BWEC-Schaferct"
    group: str = "IQL"
    name: str = "IQL"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device
        self._policy_labels = torch.zeros((buffer_size, 1), dtype=torch.long, device=device)

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        #print(f"Dataset size: {n_transitions}")
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"] / b_in_Mb)
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._policy_labels[:n_transitions] = torch.tensor(data["policy_labels"][..., None], dtype=torch.long, device=self._device)
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        policy_labels = self._policy_labels[indices]
        return [states, actions, rewards, next_states, dones, policy_labels]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError

def set_seed(
    seed: int, deterministic_torch: bool = False
):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)

def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()

def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)

class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CausalLatentFactorModule(nn.Module):
    def __init__(self, input_dim=256, latent_dim=64, num_policies=10):
        super(CausalLatentFactorModule, self).__init__()
        # Latent Factor Inference Network
        self.inference_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        # Counterfactual Reconstruction Network
        self.reconstruction_net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        # Policy Discriminator Network
        self.discriminator_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_policies)
        )

    def forward(self, features):
        # Infer latent factors
        latent_factors = self.inference_net(features)
        # Reconstruct observations
        reconstructed_features = self.reconstruction_net(latent_factors)
        # Predict policy labels
        policy_logits = self.discriminator_net(latent_factors.detach())  # Detach for discriminator
        return latent_factors, reconstructed_features, policy_logits

class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action
        self.encoder0 = nn.Parameter(torch.tensor([
            1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 
            1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 
            1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5,
            1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
            1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2,
            1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2,
            1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
            1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
            1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
            1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
            1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
            1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
            1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
            1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
            1,    1,    1,    1,    1,    1,    1,    1,    1,    1
        ], dtype=torch.float32))
        self.encoder0.requires_grad_(False)

        # encoder 1
        self.encoder1 = nn.Sequential(
            # encoder 1
            nn.Linear(150, 256),
            # nn.LayerNorm(256),
            nn.ReLU()
        )
        # GRU
        self.gru = nn.GRU(256, 256, 2)
        # FC
        self.fc_mid = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )
        # Recisual Block 1(rb1)
        self.rb1 = nn.Sequential(
            nn.Linear(320, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU()
        )
        # Recisual Block 2(rb2)
        self.rb2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU()
        )
        # final 'gmm'
        self.final = nn.Sequential(
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, obs: torch.Tensor, latent_factors: torch.Tensor, h, c):
        # Ensure obs has shape [batch_size, state_dim]
        if len(obs.shape) == 3:
            obs = obs.squeeze(1)  # Remove time dimension if present
        
        # Process observations
        obs_ = obs * self.encoder0
        features = self.encoder1(obs_)
        features = features.unsqueeze(0)  # Add time dimension for GRU: [1, batch_size, hidden_dim]
        features, _ = self.gru(features)
        features = features.squeeze(0)  # Remove time dimension: [batch_size, hidden_dim]
        
        mem1 = features
        
        # Reshape latent factors if needed
        if len(latent_factors.shape) == 3:  # [1, batch_size, latent_dim]
            latent_factors = latent_factors.squeeze(0)  # [batch_size, latent_dim]
        
        # Concatenate features and latent factors
        features_cat = torch.cat([features, latent_factors], dim=1)  # [batch_size, hidden_dim + latent_dim]
        
        # Pass through residual blocks
        features = self.rb1(features_cat) + mem1
        mem2 = features
        features = self.rb2(features) + mem2
        
        # Get mean and std
        mean = self.final(features)
        mean = mean * self.max_action * 1e6
        mean = mean.clamp(min=1)  # Lower minimum to 1 Mbps instead of removing entirely
        
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))  # [1]
        std = std.expand(mean.shape[0], 1)  # [batch_size, 1]
        
        # Concatenate mean and std
        ret = torch.cat((mean, std), dim=1)  # [batch_size, 2]
        ret = ret.unsqueeze(0)  # Add time dimension back: [1, batch_size, 2]
        
        return ret, h, c

class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )

class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)
        self.encoder = nn.Parameter(torch.tensor([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 
                                 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 
                                 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 
                                 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1
                                 ], dtype=torch.float32))
        self.encoder.requires_grad_(False)

    def both(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        state_ = state * self.encoder
        sa = torch.cat([state_, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))

class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)
        self.encoder = nn.Parameter(torch.tensor([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 
                                 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 
                                 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 
                                 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1
                                 ], dtype=torch.float32))
        self.encoder.requires_grad_(False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state_ = state * self.encoder
        return self.v(state_)


class ImplicitQLearning:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
        actor_lr: float = 2e-4,
    ):
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau

        self.total_it = 0
        self.device = device
        self.causal_module = CausalLatentFactorModule(input_dim=256, latent_dim=64, num_policies=10).to(device)
        self.causal_optimizer = torch.optim.AdamW(self.causal_module.parameters(), lr=1e-3)
        self.discriminator_optimizer = torch.optim.AdamW(self.causal_module.discriminator_net.parameters(), lr=1e-3)
        # Add error thresholds
        self.error_thresholds = {
            'small': 0.3,  # 20% error
            'medium': 0.6,  # 50% error
            'large': 1.0    # 100% error
        }
        self.error_penalties = {
            'small': 1.0,   # normal weight
            'medium': 1.5,  # 2x penalty
            'large': 2.0    # 4x penalty
        }
        # Add these lines
        self.warmup_steps = 2000
        self.base_actor_lr = actor_lr
        self.current_lr = 0
        # Add penalty factors for over/under estimation
        self.over_estimation_penalty = 2.5  # Stronger penalty for overestimation
        self.under_estimation_penalty = 1.2  # Milder penalty for underestimation

    def _compute_error_weights(self, pred, target):
        """Calculate error weights with explicit over/under estimation penalties"""
        relative_error = torch.abs(pred - target) / (target + 1e-8)
        
        # Start with base weights
        weights = torch.ones_like(relative_error)
        
        # Add asymmetric penalties for over/under estimation
        over_mask = (pred > target).float()
        under_mask = (pred < target).float()
        
        # Apply penalties based on how much over/under the prediction is
        over_amount = torch.maximum((pred - target) / (target + 1e-8), torch.zeros_like(pred))
        under_amount = torch.maximum((target - pred) / (target + 1e-8), torch.zeros_like(pred))
        
        weights = weights + (over_mask * over_amount * self.over_estimation_penalty) + \
                          (under_mask * under_amount * self.under_estimation_penalty)
        
        # Apply existing threshold penalties
        for threshold, penalty in zip(['small', 'medium', 'large'], 
                                   ['small', 'medium', 'large']):
            alpha = torch.clamp((relative_error - self.error_thresholds[threshold]) / 0.1,
                              0.0, 1.0)
            weights = weights + alpha * (self.error_penalties[penalty] - weights)
        
        return weights

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        
        # Apply error penalties
        error_weights = self._compute_error_weights(qs[0], targets)
        weighted_losses = [F.mse_loss(q, targets, reduction='none') * error_weights for q in qs]
        q_loss = sum(torch.mean(loss) for loss in weighted_losses) / len(qs)
        
        log_dict["q_score"] = (torch.mean(qs[0]).item() + torch.mean(qs[1]).item()) / 2
        log_dict["q_loss"] = q_loss.item()
        log_dict["large_error_ratio"] = torch.mean((error_weights > self.error_penalties['medium']).float()).item()
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,  # [batch_size, state_dim]
        actions: torch.Tensor,
        latent_factors: torch.Tensor,  # [batch_size, latent_dim]
        log_dict: Dict,
    ):
        # Debug shapes
        #print(f"observations shape: {observations.shape}")
        #print(f"latent_factors shape: {latent_factors.shape}")
        
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        
        # Initialize hidden states with correct batch size
        batch_size = observations.shape[0]
        h = torch.zeros((1, batch_size, 1), device=observations.device)
        c = torch.zeros((1, batch_size, 1), device=observations.device)
        
        # Get policy output
        out_, _, _ = self.actor(observations, latent_factors, h, c)  # [1, batch_size, 2]
        out_ = out_.squeeze(0)  # [batch_size, 2]
        
        mean = out_[:, 0]  # [batch_size]
        mean = mean / 1e6
        std = out_[:, 1]  # [batch_size]
        
        policy_out = Normal(mean, std)
        
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions.squeeze(-1))
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape mismatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        
        # Add error penalties to policy loss
        error_weights = self._compute_error_weights(mean, actions.squeeze(-1))
        weighted_bc_losses = bc_losses * error_weights
        
        policy_loss = torch.mean(exp_adv * weighted_bc_losses)
        log_dict["actor_all_loss"] = policy_loss.item()
        log_dict["mean_error_weight"] = torch.mean(error_weights).item()
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            policy_labels
        ) = batch
        log_dict = {}

        # Add debugging prints
        #print(f"Policy labels require grad: {policy_labels.requires_grad}")

        # Extract features
        obs_ = torch.squeeze(observations, 0)
        obs_ = obs_ * self.actor.encoder0
        features = self.actor.encoder1(obs_)
        features, _ = self.actor.gru(features)
        features = self.actor.fc_mid(features)

        # Debug features
        #print(f"Features require grad: {features.requires_grad}")

        # Get causal module outputs
        latent_factors, reconstructed_features, policy_logits = self.causal_module(features)
        
        # Debug outputs
        #print(f"Policy logits require grad: {policy_logits.requires_grad}")

        # 1. Update discriminator - Fix gradient computation
        policy_logits = policy_logits.to(torch.float32)  # Ensure correct dtype
        policy_labels = policy_labels.squeeze().to(torch.long)  # Ensure correct dtype
        
        # Ensure policy_logits requires gradient
        if not policy_logits.requires_grad:
            policy_logits.requires_grad_(True)
        
        discriminator_loss = F.cross_entropy(policy_logits, policy_labels)
        log_dict["discriminator_loss"] = discriminator_loss.item()
        
        # Check if loss requires gradient
        #print(f"Discriminator loss requires grad: {discriminator_loss.requires_grad}")
        
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward(retain_graph=True)  # Add retain_graph=True
        torch.nn.utils.clip_grad_norm_(self.causal_module.parameters(), max_norm=1.0)
        self.discriminator_optimizer.step()

        # 2. Update causal module
        latent_factors, reconstructed_features, policy_logits = self.causal_module(features.detach())
        adversarial_loss = -F.cross_entropy(policy_logits, policy_labels.squeeze())
        reconstruction_loss = F.mse_loss(reconstructed_features, features.detach())
        
        total_causal_loss = adversarial_loss + reconstruction_loss
        log_dict["adversarial_loss"] = adversarial_loss.item()
        log_dict["reconstruction_loss"] = reconstruction_loss.item()
        
        self.causal_optimizer.zero_grad()
        total_causal_loss.backward()
        self.causal_optimizer.step()

        # 3. Get fresh latent factors for policy update
        with torch.no_grad():
            latent_factors = self.causal_module.inference_net(features)

        # Update value function
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)

        # Update Q function
        with torch.no_grad():
            next_v = self.vf(next_observations)
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)

        # Update actor
        self._update_policy(adv, observations, actions, latent_factors, log_dict)

        # Log variance of latent factors
        latent_factor_variance = torch.var(latent_factors, dim=0).mean()
        log_dict["latent_factor_variance"] = latent_factor_variance.item()

        # Add warmup schedule
        if self.total_it < self.warmup_steps:
            self.current_lr = self.base_actor_lr * (self.total_it / self.warmup_steps)
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = self.current_lr
            
            # Log warmup progress
            if USE_WANDB:
                wandb.log({"learning_rate": self.current_lr}, step=self.total_it)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = state_dict["total_it"]


def get_input_from_file():
    # dummy -> real input
    #evaluation_file = '../evaluation/data/02560.json'
    evaluation_file = os.path.join(project_root_path, 'ALLdatasets', 'EMU', 'emulated_dataset_chunk_0', '02560.json')

    with open(evaluation_file, "r") as file:
        call_data = json.load(file)
    observations = np.asarray(call_data['observations'], dtype=np.float32)
    observations = observations.reshape(1, -1, STATE_DIM)
    return observations

def export2onnx(pt_path, onnx_path):
    """
    trans pt to onnx
    """
    BS = 1  # batch size
    hidden_size = 1  # number of hidden units in the LSTM
    latent_dim = 64  # dimension of latent factors

    # instantiate the ML BW estimator
    torchBwModel = GaussianPolicy(STATE_DIM, ACTION_DIM, MAX_ACTION)
    torchBwModel.load_state_dict(torch.load(pt_path))
    
    # create inputs: 1 episode x T timesteps x obs_dim features
    dummy_inputs = get_input_from_file()
    torch_dummy_inputs = torch.as_tensor(dummy_inputs)
    
    # Reshape inputs to match expected dimensions
    # From [1, T, state_dim] to [T, 1, state_dim]
    torch_dummy_inputs = torch_dummy_inputs.squeeze(0).unsqueeze(1)
    
    torch_initial_hidden_state = torch.zeros((BS, hidden_size))
    torch_initial_cell_state = torch.zeros((BS, hidden_size))
    
    # Create dummy latent factors with correct shape
    dummy_latent_factors = torch.zeros((1, latent_dim))  # [batch_size, latent_dim]
    
    # predict dummy outputs
    dummy_outputs, final_hidden_state, final_cell_state = torchBwModel(
        torch_dummy_inputs[0:1],  # Take only first timestep [1, 1, state_dim]
        dummy_latent_factors,
        torch_initial_hidden_state,
        torch_initial_cell_state
    )
    
    # save onnx model
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    torchBwModel.to("cpu")
    torchBwModel.eval()
    
    # Export with all required inputs
    torch.onnx.export(
        torchBwModel,
        (
            torch_dummy_inputs[0:1],  # [1, 1, state_dim]
            dummy_latent_factors,     # [1, latent_dim]
            torch_initial_hidden_state,
            torch_initial_cell_state
        ),
        onnx_path,
        opset_version=11,
        input_names=['obs', 'latent_factors', 'hidden_states', 'cell_states'],
        output_names=['output', 'state_out', 'cell_out'],
    )
    
    # verify torch and onnx models outputs
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    onnx_hidden_state = np.zeros((1, hidden_size), dtype=np.float32)
    onnx_cell_state = np.zeros((1, hidden_size), dtype=np.float32)
    onnx_latent_factors = np.zeros((1, latent_dim), dtype=np.float32)
    
    torch_hidden_state = torch.as_tensor(onnx_hidden_state)
    torch_cell_state = torch.as_tensor(onnx_cell_state)
    torch_latent_factors = torch.as_tensor(onnx_latent_factors)
    
    # online interaction: step through the environment 1 time step at a time
    with torch.no_grad():
        for i in tqdm(range(dummy_inputs.shape[1]), desc="Verifying  "):
            # Reshape input for current timestep
            current_input = torch_dummy_inputs[i:i+1]  # [1, 1, state_dim]
            
            torch_estimate, torch_hidden_state, torch_cell_state = torchBwModel(
                current_input,
                torch_latent_factors,
                torch_hidden_state,
                torch_cell_state
            )
            
            feed_dict = {
                'obs': current_input.numpy(),
                'latent_factors': onnx_latent_factors,
                'hidden_states': onnx_hidden_state,
                'cell_states': onnx_cell_state
            }
            
            onnx_estimate, onnx_hidden_state, onnx_cell_state = ort_session.run(None, feed_dict)
            
            assert np.allclose(torch_estimate.numpy(), onnx_estimate, atol=10), \
                'Failed to match model outputs!, {}, {}'.format(torch_estimate.numpy(), onnx_estimate)
            assert np.allclose(torch_hidden_state, onnx_hidden_state, atol=1e-7), \
                'Failed to match hidden state'
            assert np.allclose(torch_cell_state, onnx_cell_state, atol=1e-7), \
                'Failed to match cell state'
        
        assert np.allclose(torch_hidden_state, final_hidden_state, atol=1e-7), \
            'Failed to match final hidden state'
        assert np.allclose(torch_cell_state, final_cell_state, atol=1e-7), \
            'Failed to match final cell state'

def get_over_estimation_rate(pred, true):
    """Calculate overestimation rate considering magnitudes"""
    over_est = np.maximum((pred - true) / true, 0)  # Only positive differences
    return np.mean(over_est)  # Average overestimation magnitude

def get_under_estimation_rate(pred, true):
    """Calculate underestimation rate considering magnitudes"""
    under_est = np.maximum((true - pred) / true, 0)  # Only positive differences
    return np.mean(under_est)  # Average underestimation magnitude

def _evaluate_single_file(onnx_path, f_path, latent_dim):
    # Create a local InferenceSession for each process
    local_session = ort.InferenceSession(onnx_path)
    with open(f_path, 'r') as file:
        call_data = json.load(file)
    observations = np.asarray(call_data['observations'], dtype=np.float32)
    true_capacity = np.asarray(call_data['true_capacity'], dtype=np.float32) / 1e6

    model_predictions = []
    hidden_state = np.zeros((1, 1), dtype=np.float32)
    cell_state = np.zeros((1, 1), dtype=np.float32)
    latent_factors = np.zeros((1, latent_dim), dtype=np.float32)

    for t in range(observations.shape[0]):
        obss = observations[t:t+1,:].reshape(1,1,-1)
        feed_dict = {
            'obs': obss,
            'latent_factors': latent_factors,
            'hidden_states': hidden_state,
            'cell_states': cell_state
        }
        bw_prediction, hidden_state, cell_state = local_session.run(None, feed_dict)
        model_predictions.append(bw_prediction[0,0,0]/1e6)

    valid_count = 0
    call_mse = []
    call_accuracy = []
    overest_count = 0
    underest_count = 0
    for true_bw, pre_bw in zip(true_capacity, model_predictions):
        if not np.isnan(true_bw) and not np.isnan(pre_bw):
            mse_ = (true_bw - pre_bw)**2
            call_mse.append(mse_)
            accuracy_ = max(0, 1 - abs(pre_bw - true_bw)/true_bw)
            call_accuracy.append(accuracy_)
            valid_count += 1
            if pre_bw > true_bw:
                overest_count += 1
            elif pre_bw < true_bw:
                underest_count += 1

    if valid_count > 0:
        over_ = overest_count/valid_count
        under_ = underest_count/valid_count
    else:
        over_ = 0.0
        under_ = 0.0

    over_est_rate = get_over_estimation_rate(np.array(model_predictions), true_capacity)
    under_est_rate = get_under_estimation_rate(np.array(model_predictions), true_capacity)

    return np.mean(call_mse), np.mean(call_accuracy), over_, under_, over_est_rate, under_est_rate

def evaluate(onnx_path):
    latent_dim = 64
    every_call_mse = []
    every_call_accuracy = []
    every_call_overest = []
    every_call_underest = []
    every_call_over_est_rate = []
    every_call_under_est_rate = []
    # Parallelizing the evaluation
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(_evaluate_single_file, onnx_path, f, latent_dim) for f in small_evaluation_datasets]
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Evaluating"):
            mse_, acc_, over_, under_, over_est_rate, under_est_rate = fut.result()
            every_call_mse.append(mse_)
            every_call_accuracy.append(acc_)
            every_call_overest.append(over_)
            every_call_underest.append(under_)
            every_call_over_est_rate.append(over_est_rate)
            every_call_under_est_rate.append(under_est_rate)

    every_call_mse = np.asarray(every_call_mse, dtype=np.float32)
    every_call_accuracy = np.asarray(every_call_accuracy, dtype=np.float32)
    every_call_overest = np.asarray(every_call_overest, dtype=np.float32)
    every_call_underest = np.asarray(every_call_underest, dtype=np.float32)
    every_call_over_est_rate = np.asarray(every_call_over_est_rate, dtype=np.float32)
    every_call_under_est_rate = np.asarray(every_call_under_est_rate, dtype=np.float32)
    return (
        np.mean(every_call_mse),
        np.mean(every_call_accuracy),
        np.mean(every_call_overest),
        np.mean(every_call_underest),
        np.mean(every_call_over_est_rate),
        np.mean(every_call_under_est_rate)
    )

@pyrallis.wrap()
def train(config: TrainConfig):
    torch.autograd.set_detect_anomaly(True)  # Add at the start of the train() function
    state_dim = STATE_DIM
    action_dim = ACTION_DIM

    testdataset_file = open(pickle_path, 'rb')
    dataset = pickle.load(testdataset_file)
    print('dataset loaded')

    # Initialize replay buffer on CPU
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        device="cpu",  # Use CPU memory
    )
    replay_buffer.load_dataset(dataset)

    max_action = MAX_ACTION

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed)

    q_network = TwinQ(state_dim, action_dim).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)
    actor = (
        DeterministicPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
        if config.iql_deterministic
        else GaussianPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
    ).to(config.device)
    v_optimizer = torch.optim.AdamW(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.AdamW(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=config.actor_lr)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.max_timesteps,
        "actor_lr": config.actor_lr,
    }

    print("---------------------------------------")
    print(f"Training IQL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = ImplicitQLearning(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    if USE_WANDB:
        wandb_init(asdict(config))

    # Add gradient clipping
    torch.nn.utils.clip_grad_norm_(trainer.causal_module.parameters(), max_norm=1.0)
    
    # Add these debug prints before training
    print("Checking model parameters require gradients:")
    for name, param in trainer.causal_module.named_parameters():
        print(f"{name}: {param.requires_grad}")
    
    # Changed code: track best metrics
    best_mse = float('inf')
    best_mse_ckpt = ""
    best_err = float('inf')
    best_err_ckpt = ""
    best_over = float('inf')
    best_over_ckpt = ""
    best_under = float('inf')
    best_under_ckpt = ""
    best_over_est_rate = float('inf')
    best_over_est_rate_ckpt = ""

    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]  # Move batch to GPU
        log_dict = trainer.train(batch)
        if USE_WANDB:
            wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")

            pt_path = os.path.join(config.checkpoints_path, f"checkpoint_{t + 1}.pt")
            onnx_path = os.path.join(config.checkpoints_path, f"checkpoint_{t + 1}.onnx")
            # Save model checkpoint
            if config.checkpoints_path is not None:
                torch.save(trainer.state_dict()["actor"], pt_path)
            # Export to ONNX
            export2onnx(pt_path, onnx_path)
            # Evaluate the model
            mse_, accuracy_, over_, under_, over_est_rate, under_est_rate = evaluate(onnx_path)
            err_ = 1 - accuracy_

            # Update best metrics
            if mse_ < best_mse:
                best_mse = mse_
                best_mse_ckpt = f"checkpoint_{t+1}"
            if err_ < best_err:
                best_err = err_
                best_err_ckpt = f"checkpoint_{t+1}"
            if over_ < best_over:
                best_over = over_
                best_over_ckpt = f"checkpoint_{t+1}"
            if under_ < best_under:
                best_under = under_
                best_under_ckpt = f"checkpoint_{t+1}"
            if over_est_rate < best_over_est_rate:
                best_over_est_rate = over_est_rate
                best_over_est_rate_ckpt = f"checkpoint_{t+1}"

            print(f"[Eval @ {t+1}] Curr MSE={mse_:.4f}, Err={err_:.4f}, OverRate={over_:.4f}, UnderRate={under_:.4f}, OverEstRate={over_est_rate:.4f}, UnderEstRate={under_est_rate:.4f}")
            print(f"Best MSE so far: {best_mse:.4f} from {best_mse_ckpt}, Best Err Rate: {best_err:.4f} from {best_err_ckpt}")
            print(f"Best OverRate: {best_over:.4f} from {best_over_ckpt}, Best UnderRate: {best_under:.4f} from {best_under_ckpt}, Best OverEstRate: {best_over_est_rate:.4f} from {best_over_est_rate_ckpt}")

            if USE_WANDB and trainer.total_it > 1000:
                wandb.log({"mse": mse_, "error_rate": err_}, step=trainer.total_it)


if __name__ == "__main__":
    train()
