#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO: define random functions  
# __all__ = ['gaussin', 
#            'uniform', 
#            'shuffle',
#            'randn',
#            'rand',
#            'randint']

from ..fluid import core
from ..fluid.framework import device_guard, in_dygraph_mode, _varbase_creator
from ..fluid.layers.layer_function_generator import templatedoc
from ..fluid.layer_helper import LayerHelper
from ..fluid.data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype

__all__ = ['randperm']


@templatedoc()
def randperm(n,
             out=None,
             dtype="int64",
             device=None,
             stop_gradient=True,
             seed=0):
    """
    This operator returns a random permutation of integers from 0 to n - 1.

    Args:
        n (int): The upper bound (exclusive), and it should be greater than 0.
        out (Variable, optional): Optional output which can be any created 
            Variable that meets the requirements to store the result of operation.
            If out is None, a new Varibale will be create to store the result. 
            Default: None.
        dtype (np.dtype|core.VarDesc.VarType|str, optional): The type of the 
            output Tensor. Supported data types: int64, int32. Default: int32.
        device (str, optional): Specific the output variable to be saved in cpu
            or gpu memory. Supported None, 'cpu', 'gpu'. If it is None, the output
            variable will be automatically assigned devices.
            Default: None.
        stop_gradient (bool, optional): Whether grad should record operations 
            on the returned tensor. Default: True.
        seed (int, optional): Random seed used for permute samples. If seed is 
            equal to 0, it means use a seed generated by the system. Note that 
            if seed is not 0, this operator will always generate the same random 
            permutation every time. Default: 0.

    Returns:
        Variable: A Tensor with a random permutation of integers from 0 to n - 1,
            and the shape of returned tensor is [n].

    Raise:
        TypeError: The dtype must be one of int32 and int64, and the data type 
            of out Tensor must be the same as the dtype. 
        ValueError: The input n should be greater than 0 in randperm op.
        ValueError: The input device should in [None, 'cpu', 'gpu'].

    Examples:
        .. code-block:: python
	    import paddle
	    import paddle.fluid as fluid

	    num = 6
	    is_use_gpu = False

	    data_1 = paddle.tensor.randperm(num)
	    fluid.layers.Print(data_1)

	    data_2 = paddle.tensor.randperm(num, dtype="int32", seed=1)
	    fluid.layers.Print(data_2)

	    data_3 = paddle.tensor.randperm(num, stop_gradient=False, device="cpu")
	    fluid.layers.Print(data_3)

	    paddle.tensor.randperm(num, out=data_3)
	    fluid.layers.Print(data_3)

	    place = fluid.CUDAPlace(0) if is_use_gpu else fluid.CPUPlace()
	    exe = fluid.Executor(place)
	    exe.run(fluid.default_startup_program())
	    exe.run()
 
    """

    if n < 1:
        raise ValueError("The input n should be greater than 0 in randperm op.")
    check_dtype(dtype, 'dtype', ['int64', 'int32'], 'randperm')
    dtype = convert_dtype(dtype)
    if device not in [None, 'cpu', 'gpu']:
        raise ValueError("The input device should in [None, 'cpu', 'gpu'].")
    check_type(stop_gradient, 'stop_gradient', bool, 'randperm')

    helper = LayerHelper("randperm", **locals())
    if out is None:
        out = helper.create_variable_for_type_inference(dtype=dtype)
    else:
        check_variable_and_dtype(out, 'out', [dtype], 'randperm')
    if stop_gradient:
        out.stop_gradient = True
    inputs = dict()
    outputs = {'Out': [out]}
    attrs = {'n': n, 'dtype': out.dtype, 'seed': seed}
    with device_guard(device):
        helper.append_op(
            type='randperm', inputs=inputs, outputs=outputs, attrs=attrs)
    return out
