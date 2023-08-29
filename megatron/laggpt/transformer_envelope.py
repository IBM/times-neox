import torch
from torch import nn
from gluonts.itertools import prod
from gluonts.time_feature import get_lags_for_frequency



from .gluontstorch.utils import take_last, repeat_along_dim, lagged_sequence_values, unsqueeze_expand
from .gluontstorch.scaler import StdScaler, MeanScaler, NOPScaler
from .gluontstorch.studentt import StudentTOutput
from .gluontstorch.loss import DistributionLoss, NegativeLogLikelihood


from megatron.utils import get_attn_mask
from megatron import print_rank_0


class TransformerEnvelope:
    """
    A class that defines input scaling, batch funtion, loss and calculation of greedy vals for lag-GPT.
    Code is taken from original lag-GPT model.
    """
    def __init__(
            self, context_length, 
            scaling,
            hidden_size,
            distribution_head=StudentTOutput(),
            loss: DistributionLoss = NegativeLogLikelihood()):
        

        self.embedding_dim = hidden_size
        self.probabilistic_loss = loss

        self.context_length = context_length

        if scaling == "mean":
            print_rank_0("> mean scaling ...")
            self.scaler = MeanScaler(keepdim=True, dim=1)
        elif scaling == "std":
            self.scaler = StdScaler(keepdim=True, dim=1)
            print_rank_0("> std scaling ...")
        else:
            self.scaler = NOPScaler(keepdim=True, dim=1)

        self.lags_seq = sorted(
            list(
                set(
                    get_lags_for_frequency(freq_str="Q", num_default_lags=1)
                    + get_lags_for_frequency(freq_str="M", num_default_lags=1)
                    + get_lags_for_frequency(freq_str="W", num_default_lags=1)
                    + get_lags_for_frequency(freq_str="D", num_default_lags=1)
                    + get_lags_for_frequency(freq_str="H", num_default_lags=1)
                    + get_lags_for_frequency(freq_str="T", num_default_lags=1)
                    + get_lags_for_frequency(freq_str="S", num_default_lags=1)
                )
            )
        )
        self.dist_head = distribution_head
        n_scaling_factors = 2
        self._feature_size= len(self.lags_seq) + n_scaling_factors
    

    @property
    def distribution_projection(self):
        projection = self.dist_head.get_args_proj(self.embedding_dim).cuda()
        for param in projection.parameters():
            torch.distributed.broadcast(param, src=0)
        return projection

    @property
    def feature_size(self):
        return self._feature_size

    @property
    def modules(self):
        return self._modules

    @property
    def past_length(self) -> int:
        return self.context_length + max(self.lags_seq)
    

    def get_loc_scale(self, batch):
        past_target = batch["past_target"]
        past_observed_values = batch["past_observed_values"]
        _, loc, scale = self.scaler(past_target, past_observed_values)
        return loc, scale


    def batch_fn(self, batch):
       
        with torch.no_grad():
            past_target = batch["past_target"]
            past_observed_values = batch["past_observed_values"]
            future_target = batch["future_target"]
            future_observed_values = batch["future_observed_values"]
            
            # TODO, not supported
            extra_dims = len(future_target.shape) - len(past_target.shape)
            extra_shape = future_target.shape[:extra_dims]
            
            repeats = prod(extra_shape)
            past_target = repeat_along_dim(past_target, 0, repeats)
            past_observed_values = repeat_along_dim(past_observed_values, 0, repeats)

            future_target = future_target.reshape(
                -1,
                *future_target.shape[extra_dims + 1 :],
            )
            future_observed = future_observed_values.reshape(
                -1,
                *future_observed_values.shape[extra_dims + 1 :],
            )

            scaled_past_target, loc, scale = self.scaler(past_target, past_observed_values)
            
            if future_target is not None:
                future_length = future_target.shape[1]
                input = torch.cat(
                    (
                        scaled_past_target[..., -self.context_length :],
                        (future_target[..., : future_length - 1] - loc) / scale,
                    ),
                    dim=-1,
                )
            else:
                input = scaled_past_target[..., -self.context_length :]

            
            if future_target is not None:
                future_length = future_target.shape[1]
                input = torch.cat(
                    (
                        scaled_past_target[..., -self.context_length :],
                        (future_target[..., : future_length - 1] - loc) / scale,
                    ),
                    dim=-1,
                )
            else:
                input = scaled_past_target[..., -self.context_length :]


            prior_input = (past_target[..., : -self.context_length] - loc) / scale
            lags = lagged_sequence_values(self.lags_seq, prior_input, input, dim=-1)
            
            static_feat = torch.cat((loc.abs().log1p(), scale.log()), dim=-1)
            expanded_static_feat = unsqueeze_expand(
                static_feat, dim=-2, size=lags.shape[-2]
            )
            
            tokens = torch.cat((lags, expanded_static_feat), dim=-1)
            seq_length = tokens.shape[1]
            attn_mask = get_attn_mask(seq_length=seq_length, device=tokens.device)

            pipe_tuple = (
                (tokens, attn_mask), 
                (
                    past_target, future_target, 
                    past_observed_values, future_observed, loc, scale
                )
            )
        
            return pipe_tuple


    #past_target, future_target, past_observed_values, future_observed, loc, scale
    def loss(self, dist_args, pipe_tuple):
    
        past_target, future_target, past_observed_values, future_observed, loc, scale = pipe_tuple
        dist = self.dist_head.distribution(dist_args, loc, scale)
        
        context_target = take_last(
            past_target, dim=-1, num=self.context_length - 1
        )

        target = torch.cat(
            (context_target, future_target),
            dim=1,
        )
        
        context_observed = take_last(
            past_observed_values, dim=-1, num=self.context_length - 1
        )
        observed_values = torch.cat((context_observed, future_observed), dim=1)
        result_loss = (self.probabilistic_loss(dist, target) * observed_values).sum() / observed_values.sum().clamp_min(1.0)
        
        return result_loss


    #past_target, future_target, past_observed_values, future_observed, loc, scale
    def get_greedy_val(self, dist_args, loc, scale):
    
        dist = self.dist_head.distribution(dist_args, loc, scale)
        return dist.mean
        