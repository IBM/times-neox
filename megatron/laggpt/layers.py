import torch.nn.init as init
from megatron.mpu.layers import ColumnParallelLinear




class ProjectionPipe(ColumnParallelLinear):
    """
    Modification of ColumnParallelLinear to pass attention mask through the projection layer.
    """
    def __init__(
        self,
        neox_args,
        input_size,
        output_size,
        bias=True,
        gather_output=True,
        init_method=init.xavier_normal_,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        mup_rescale_parameters=False,
    ):
        super(ProjectionPipe, self).__init__(
            neox_args,
            input_size,
            output_size,
            bias,
            gather_output,
            init_method,
            stride,
            keep_master_weight_for_test,
            skip_bias_add,
            mup_rescale_parameters,
        )

    def forward(self, input_):

        input_, attn_mask = input_
        # Second argument is bias if skip_bias_add is not None
        output, _ = super(ProjectionPipe, self).forward(input_)
        return output, attn_mask

    

    