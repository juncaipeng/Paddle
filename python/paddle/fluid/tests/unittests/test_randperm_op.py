#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.op import Operator
from paddle.fluid import Program, program_guard


def check_randperm_out(n, data_np):
    assert isinstance(data_np, np.ndarray), \
        "The input data_np should be np.ndarray."
    gt_sorted = np.arange(n)
    out_sorted = np.sort(data_np)
    return list(gt_sorted == out_sorted)


def convert_dtype(dtype_str):
    dtype_str_list = ["int32", "int64"]
    dtype_num_list = [2, 3]
    assert dtype_str in dtype_str_list, dtype_str + \
        " should in " + str(dtype_str_list)
    return dtype_num_list[dtype_str_list.index(dtype_str)]


class TestUniformRandomOp(OpTest):
    def setUp(self):
        self.op_type = "randperm"
        self.n = 200
        self.dtype = "int64"
        self.force_cpu = False
        self.seed = 0

        self.inputs = {}
        self.outputs = {"Out": np.zeros((self.n)).astype(self.dtype)}
        self.init_attrs()
        self.attrs = {
            "n": self.n,
            "dtype": convert_dtype(self.dtype),
            "force_cpu": self.force_cpu,
            "seed": self.seed,
        }

    def init_attrs(self):
        pass

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        out_np = np.array(outs[0])
        self.assertTrue(check_randperm_out(self.n, out_np),
            msg="The sorted ground truth and sorted out should " + \
                "be equal, out = " + str(out_np))


class TestUniformRandomOp_attr_n(TestUniformRandomOp):
    def init_attrs(self):
        self.n = 10000


class TestUniformRandomOp_attr_int32(TestUniformRandomOp):
    def init_attrs(self):
        self.dtype = "int32"


class TestUniformRandomOp_attr_force_cpu(TestUniformRandomOp):
    def init_attrs(self):
        self.force_cpu = True


class TestUniformRandomOp_attr_seed(TestUniformRandomOp):
    def init_attrs(self):
        self.seed = 10


class TestRandpermOpError(unittest.TestCase):
    def test_errors(self):
        main_prog = Program()
        start_prog = Program()
        with program_guard(main_prog, start_prog):

            def test_Variable():
                out = np.arange(10)
                paddle.tensor.randperm(n=10, out=out)

            self.assertRaises(TypeError, test_Variable)

            def test_value():
                paddle.tensor.randperm(n=-3)

            self.assertRaises(ValueError, test_value)


class TestUniformRandomOp_attr_out(unittest.TestCase):
    def test_attr_tensor_API(self):
        startup_program = fluid.Program()
        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            n = 10
            data_1 = fluid.layers.fill_constant([n], "int64", 3)
            paddle.tensor.randperm(n=n, out=data_1)

            data_2 = paddle.tensor.randperm(n=n, dtype="int32", force_cpu=True)

            place = fluid.CPUPlace()
            if fluid.core.is_compiled_with_cuda():
                place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)

            exe.run(startup_program)
            outs = exe.run(train_program, fetch_list=[data_1, data_2])

            out_np = np.array(outs[0])
            self.assertTrue(check_randperm_out(n, out_np),
                msg="The sorted ground truth and sorted out should " + \
                    "be equal, out = " + str(out_np))


if __name__ == "__main__":
    unittest.main()
