"""
BLADE unit test suite
=====================
Run from the repo root:
    python tests/test_blade.py
"""

import sys, math, unittest
sys.path.insert(0, ".")
import blade
import blade.nn   as nn
import blade.optim as optim
import blade.data  as data

TOL = 1e-5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def T(data, shape, grad=False):
    t = blade.Tensor.from_data(shape, data)
    t.requires_grad = grad
    return t

def allclose(a, b, tol=TOL):
    a, b = list(a), list(b)
    return len(a) == len(b) and all(abs(x - y) < tol for x, y in zip(a, b))

def ref_softmax(xs):
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s  = sum(es)
    return [e / s for e in es]

def ref_log_softmax(xs):
    m = max(xs)
    log_sum = math.log(sum(math.exp(x - m) for x in xs))
    return [x - m - log_sum for x in xs]


# ===========================================================================
# 1. Tensor factories
# ===========================================================================

class TestTensorFactories(unittest.TestCase):

    def test_zeros(self):
        t = blade.Tensor.zeros([2, 3])
        self.assertEqual(list(t.shape), [2, 3])
        self.assertTrue(allclose(t.storage(), [0.0] * 6))

    def test_ones(self):
        t = blade.Tensor.ones([3])
        self.assertTrue(allclose(t.storage(), [1.0, 1.0, 1.0]))

    def test_full(self):
        self.assertTrue(allclose(blade.Tensor.full([2, 2], 7.0).storage(), [7.0] * 4))

    def test_from_data_roundtrip(self):
        data_in = [1.0, 2.0, 3.0, 4.0]
        t = T(data_in, [2, 2])
        self.assertEqual(list(t.shape), [2, 2])
        self.assertTrue(allclose(t.storage(), data_in))

    def test_arange(self):
        self.assertTrue(allclose(blade.Tensor.arange(0, 5, 1).storage(), [0, 1, 2, 3, 4]))

    def test_randn_shape(self):
        t = blade.Tensor.randn([4, 5])
        self.assertEqual(list(t.shape), [4, 5])
        self.assertEqual(t.numel, 20)

    def test_uniform_bounds(self):
        vals = blade.Tensor.uniform([1000], 0.0, 1.0).storage()
        self.assertTrue(all(0.0 <= v <= 1.0 for v in vals))

    def test_ndim_and_numel(self):
        t = T([1, 2, 3, 4, 5, 6], [2, 3])
        self.assertEqual(t.ndim, 2)
        self.assertEqual(t.numel, 6)


# ===========================================================================
# 2. Element-wise ops — forward
# ===========================================================================

class TestElementWiseForward(unittest.TestCase):

    def setUp(self):
        self.a = T([1.0, 2.0, 3.0], [3])
        self.b = T([4.0, 5.0, 6.0], [3])

    def test_add_tensor(self):
        self.assertTrue(allclose((self.a + self.b).storage(), [5, 7, 9]))

    def test_sub_tensor(self):
        self.assertTrue(allclose((self.a - self.b).storage(), [-3, -3, -3]))

    def test_mul_tensor(self):
        self.assertTrue(allclose((self.a * self.b).storage(), [4, 10, 18]))

    def test_div_tensor(self):
        self.assertTrue(allclose((self.a / self.b).storage(), [0.25, 0.4, 0.5]))

    def test_neg(self):
        self.assertTrue(allclose(blade.ops.neg(self.a).storage(), [-1, -2, -3]))

    def test_add_scalar(self):
        self.assertTrue(allclose((self.a + 10.0).storage(), [11, 12, 13]))

    def test_mul_scalar(self):
        self.assertTrue(allclose((self.a * 2.0).storage(), [2, 4, 6]))

    def test_pow(self):
        self.assertTrue(allclose(blade.ops.pow(self.a, 2.0).storage(), [1, 4, 9]))

    def test_exp(self):
        expected = [math.exp(x) for x in [1, 2, 3]]
        self.assertTrue(allclose(blade.ops.exp(self.a).storage(), expected))

    def test_log(self):
        expected = [math.log(x) for x in [1, 2, 3]]
        self.assertTrue(allclose(blade.ops.log(self.a).storage(), expected))

    def test_sqrt(self):
        self.assertTrue(allclose(blade.ops.sqrt(T([1.0, 4.0, 9.0], [3])).storage(), [1, 2, 3]))

    def test_abs(self):
        self.assertTrue(allclose(blade.ops.abs(T([-1.0, 2.0, -3.0], [3])).storage(), [1, 2, 3]))

    def test_clamp(self):
        self.assertTrue(allclose(
            blade.ops.clamp(T([-1.0, 0.5, 2.0], [3]), 0.0, 1.0).storage(), [0, 0.5, 1]))


# ===========================================================================
# 3. Element-wise ops — backward
# ===========================================================================

class TestElementWiseBackward(unittest.TestCase):

    def test_add_grad(self):
        a = T([1.0, 2.0], [2], grad=True)
        b = T([3.0, 4.0], [2], grad=True)
        blade.ops.sum(a + b).backward()
        self.assertTrue(allclose(a.grad.storage(), [1, 1]))
        self.assertTrue(allclose(b.grad.storage(), [1, 1]))

    def test_mul_grad(self):
        a = T([2.0, 3.0], [2], grad=True)
        b = T([4.0, 5.0], [2], grad=True)
        blade.ops.sum(a * b).backward()
        self.assertTrue(allclose(a.grad.storage(), [4, 5]))
        self.assertTrue(allclose(b.grad.storage(), [2, 3]))

    def test_pow_grad(self):
        a = T([1.0, 2.0, 3.0], [3], grad=True)
        blade.ops.sum(blade.ops.pow(a, 3.0)).backward()
        self.assertTrue(allclose(a.grad.storage(), [3 * x**2 for x in [1, 2, 3]]))

    def test_exp_grad(self):
        a = T([0.0, 1.0], [2], grad=True)
        blade.ops.sum(blade.ops.exp(a)).backward()
        self.assertTrue(allclose(a.grad.storage(), [math.exp(x) for x in [0, 1]]))

    def test_log_grad(self):
        a = T([1.0, 2.0, 4.0], [3], grad=True)
        blade.ops.sum(blade.ops.log(a)).backward()
        self.assertTrue(allclose(a.grad.storage(), [1.0, 0.5, 0.25]))

    def test_neg_grad(self):
        a = T([1.0, 2.0], [2], grad=True)
        blade.ops.sum(blade.ops.neg(a)).backward()
        self.assertTrue(allclose(a.grad.storage(), [-1, -1]))

    def test_chain_relu_sum(self):
        a = T([1.0, -2.0, 3.0], [3], grad=True)
        blade.ops.sum(blade.ops.relu(a)).backward()
        self.assertTrue(allclose(a.grad.storage(), [1, 0, 1]))

    def test_chain_exp_mul(self):
        a = T([0.0, 1.0], [2], grad=True)
        blade.ops.sum(blade.ops.exp(a * 2.0)).backward()
        self.assertTrue(allclose(a.grad.storage(), [2 * math.exp(2 * x) for x in [0, 1]]))


# ===========================================================================
# 4. Global reductions
# ===========================================================================

class TestGlobalReductions(unittest.TestCase):

    def test_sum_forward(self):
        self.assertAlmostEqual(blade.ops.sum(T([1, 2, 3, 4], [4])).item(), 10.0)

    def test_sum_backward(self):
        a = T([1.0, 2.0, 3.0], [3], grad=True)
        blade.ops.sum(a).backward()
        self.assertTrue(allclose(a.grad.storage(), [1, 1, 1]))

    def test_mean_forward(self):
        self.assertAlmostEqual(blade.ops.mean(T([1, 2, 3, 4], [4])).item(), 2.5)

    def test_mean_backward(self):
        a = T([1.0, 2.0, 3.0, 4.0], [4], grad=True)
        blade.ops.mean(a).backward()
        self.assertTrue(allclose(a.grad.storage(), [0.25] * 4))

    def test_max_forward(self):
        self.assertAlmostEqual(blade.ops.max(T([3, 1, 4, 1, 5], [5])).item(), 5.0)

    def test_max_backward(self):
        a = T([3.0, 1.0, 4.0, 1.0, 5.0], [5], grad=True)
        blade.ops.max(a).backward()
        self.assertTrue(allclose(a.grad.storage(), [0, 0, 0, 0, 1]))


# ===========================================================================
# 5. Dim-wise reductions
# ===========================================================================

class TestDimReductions(unittest.TestCase):
    DATA  = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    SHAPE = [2, 3]

    def test_sum_dim0_shape(self):
        self.assertEqual(list(blade.ops.sum(T(self.DATA, self.SHAPE), 0).shape), [3])

    def test_sum_dim0_vals(self):
        self.assertTrue(allclose(blade.ops.sum(T(self.DATA, self.SHAPE), 0).storage(), [5, 7, 9]))

    def test_sum_dim1_vals(self):
        self.assertTrue(allclose(blade.ops.sum(T(self.DATA, self.SHAPE), 1).storage(), [6, 15]))

    def test_sum_keepdim(self):
        self.assertEqual(list(blade.ops.sum(T(self.DATA, self.SHAPE), 0, True).shape), [1, 3])

    def test_sum_negative_dim(self):
        self.assertTrue(allclose(blade.ops.sum(T(self.DATA, self.SHAPE), -1).storage(), [6, 15]))

    def test_sum_dim0_backward(self):
        a = T(self.DATA, self.SHAPE, grad=True)
        blade.ops.sum(blade.ops.sum(a, 0)).backward()
        self.assertTrue(allclose(a.grad.storage(), [1.0] * 6))

    def test_sum_dim1_backward(self):
        a = T(self.DATA, self.SHAPE, grad=True)
        blade.ops.sum(blade.ops.sum(a, 1)).backward()
        self.assertTrue(allclose(a.grad.storage(), [1.0] * 6))

    def test_mean_dim0_vals(self):
        self.assertTrue(allclose(blade.ops.mean(T(self.DATA, self.SHAPE), 0).storage(), [2.5, 3.5, 4.5]))

    def test_mean_dim1_vals(self):
        self.assertTrue(allclose(blade.ops.mean(T(self.DATA, self.SHAPE), 1).storage(), [2.0, 5.0]))

    def test_mean_dim0_backward(self):
        a = T(self.DATA, self.SHAPE, grad=True)
        blade.ops.sum(blade.ops.mean(a, 0)).backward()
        self.assertTrue(allclose(a.grad.storage(), [0.5] * 6))

    def test_max_dim0_vals(self):
        self.assertTrue(allclose(blade.ops.max(T(self.DATA, self.SHAPE), 0).storage(), [4, 5, 6]))

    def test_max_dim1_vals(self):
        self.assertTrue(allclose(blade.ops.max(T(self.DATA, self.SHAPE), 1).storage(), [3, 6]))

    def test_max_dim0_backward(self):
        a = T(self.DATA, self.SHAPE, grad=True)
        blade.ops.sum(blade.ops.max(a, 0)).backward()
        self.assertTrue(allclose(a.grad.storage(), [0, 0, 0, 1, 1, 1]))

    def test_max_dim1_backward(self):
        a = T(self.DATA, self.SHAPE, grad=True)
        blade.ops.sum(blade.ops.max(a, 1)).backward()
        self.assertTrue(allclose(a.grad.storage(), [0, 0, 1, 0, 0, 1]))

    def test_max_ties_first_wins(self):
        a = T([5.0, 5.0, 5.0], [1, 3], grad=True)
        blade.ops.max(a, 1).backward()
        self.assertTrue(allclose(a.grad.storage(), [1, 0, 0]))


# ===========================================================================
# 6. Argmax
# ===========================================================================

class TestArgmax(unittest.TestCase):

    def test_argmax_dim1_basic(self):
        # [[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]] -> [1, 0]
        t = T([0.1, 0.9, 0.0, 0.8, 0.1, 0.1], [2, 3])
        self.assertTrue(allclose(blade.ops.argmax(t, 1).storage(), [1, 0]))

    def test_argmax_dim0(self):
        # col-wise max: [[1,5],[3,2]] -> [1, 0]
        t = T([1.0, 5.0, 3.0, 2.0], [2, 2])
        self.assertTrue(allclose(blade.ops.argmax(t, 0).storage(), [1, 0]))

    def test_argmax_output_shape(self):
        t = T([0.0] * 12, [3, 4])
        self.assertEqual(list(blade.ops.argmax(t, 1).shape), [3])

    def test_argmax_no_grad(self):
        t = T([1.0, 2.0, 3.0], [1, 3], grad=True)
        out = blade.ops.argmax(t, 1)
        self.assertFalse(out.requires_grad)


# ===========================================================================
# 7. Matmul
# ===========================================================================

class TestMatmul(unittest.TestCase):

    def test_2d_shape(self):
        self.assertEqual(list(blade.ops.matmul(T([1]*6, [2, 3]), T([1]*12, [3, 4])).shape), [2, 4])

    def test_2d_values(self):
        a = T([1,2,3,4], [2,2]); b = T([5,6,7,8], [2,2])
        self.assertTrue(allclose(blade.ops.matmul(a, b).storage(), [19, 22, 43, 50]))

    def test_batched_shape(self):
        self.assertEqual(list(blade.ops.matmul(T([1]*12, [2,2,3]), T([1]*12, [2,3,2])).shape), [2, 2, 2])

    def test_matmul_backward(self):
        a = T([1,2,3,4], [2,2], grad=True)
        b = T([1,0,0,1], [2,2], grad=True)  # identity
        blade.ops.sum(blade.ops.matmul(a, b)).backward()
        self.assertTrue(allclose(a.grad.storage(), [1]*4))


# ===========================================================================
# 8. Activations
# ===========================================================================

class TestActivations(unittest.TestCase):

    def test_relu_forward(self):
        self.assertTrue(allclose(blade.ops.relu(T([-2.0, 0.0, 3.0], [3])).storage(), [0, 0, 3]))

    def test_relu_backward(self):
        a = T([-1.0, 2.0, -3.0, 4.0], [4], grad=True)
        blade.ops.sum(blade.ops.relu(a)).backward()
        self.assertTrue(allclose(a.grad.storage(), [0, 1, 0, 1]))

    def test_leaky_relu_forward(self):
        self.assertTrue(allclose(blade.ops.leaky_relu(T([-2.0, 3.0], [2]), 0.1).storage(), [-0.2, 3.0]))

    def test_sigmoid_forward(self):
        self.assertAlmostEqual(blade.ops.sigmoid(T([0.0], [1])).item(), 0.5, places=5)

    def test_sigmoid_backward(self):
        a = T([0.0], [1], grad=True)
        blade.ops.sigmoid(a).backward()
        self.assertAlmostEqual(a.grad.storage()[0], 0.25, places=5)

    def test_tanh_forward(self):
        self.assertAlmostEqual(blade.ops.tanh(T([0.0], [1])).item(), 0.0, places=5)

    def test_tanh_backward(self):
        a = T([0.0], [1], grad=True)
        blade.ops.tanh(a).backward()
        self.assertAlmostEqual(a.grad.storage()[0], 1.0, places=5)

    def test_softmax_sums_to_one(self):
        self.assertAlmostEqual(sum(blade.ops.softmax(T([1.0, 2.0, 3.0], [1, 3]), 1).storage()), 1.0, places=5)

    def test_softmax_values(self):
        logits = [1.0, 2.0, 3.0]
        self.assertTrue(allclose(blade.ops.softmax(T(logits, [1, 3]), 1).storage(), ref_softmax(logits)))

    def test_softmax_numerically_stable(self):
        self.assertAlmostEqual(sum(blade.ops.softmax(T([1000.0, 1001.0, 1002.0], [1, 3]), 1).storage()), 1.0, places=5)

    def test_softmax_backward_ones_upstream(self):
        a = T([1.0, 2.0, 3.0], [1, 3], grad=True)
        blade.ops.softmax(a, 1).backward()
        self.assertTrue(allclose(a.grad.storage(), [0.0, 0.0, 0.0]))

    def test_softmax_backward_selective(self):
        logits = [1.0, 2.0, 3.0]; s = ref_softmax(logits)
        a = T(logits, [1, 3], grad=True)
        blade.ops.sum(blade.ops.softmax(a, 1) * T([1.0, 0.0, 0.0], [1, 3])).backward()
        expected = [s[i] * ((1.0 if i == 0 else 0.0) - s[0]) for i in range(3)]
        self.assertTrue(allclose(a.grad.storage(), expected))

    def test_log_softmax_values(self):
        logits = [1.0, 2.0, 3.0]
        self.assertTrue(allclose(blade.ops.log_softmax(T(logits, [1, 3]), 1).storage(), ref_log_softmax(logits)))

    def test_log_softmax_equals_log_of_softmax(self):
        logits = [0.5, 1.5, -0.5]
        lsm = blade.ops.log_softmax(T(logits, [1, 3]), 1).storage()
        sm  = [math.log(x) for x in ref_softmax(logits)]
        self.assertTrue(allclose(lsm, sm))

    def test_log_softmax_numerically_stable(self):
        vals = blade.ops.log_softmax(T([1000.0, 1001.0, 1002.0], [1, 3]), 1).storage()
        self.assertTrue(all(abs(v) < 10 for v in vals))

    def test_log_softmax_backward_sums_to_zero(self):
        a = T([1.0, 2.0, 3.0], [1, 3], grad=True)
        blade.ops.log_softmax(a, 1).backward()
        self.assertAlmostEqual(sum(a.grad.storage()), 0.0, places=5)

    def test_log_softmax_backward_values(self):
        logits = [1.0, 2.0, 3.0]; s = ref_softmax(logits)
        a = T(logits, [1, 3], grad=True)
        blade.ops.log_softmax(a, 1).backward()
        self.assertTrue(allclose(a.grad.storage(), [1.0 - s[i] * 3 for i in range(3)]))

    def test_softmax_batched(self):
        t = T([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], [2, 3])
        out = blade.ops.softmax(t, 1)
        self.assertTrue(allclose(out.storage()[:3], ref_softmax([1, 2, 3])))
        self.assertTrue(allclose(out.storage()[3:], ref_softmax([0, 0, 0])))


# ===========================================================================
# 9. Shape ops
# ===========================================================================

class TestShapeOps(unittest.TestCase):

    def test_reshape(self):
        r = T(list(range(6)), [6]).reshape([2, 3])
        self.assertEqual(list(r.shape), [2, 3])
        self.assertTrue(allclose(r.storage(), list(range(6))))

    def test_flatten(self):
        self.assertEqual(list(T(list(range(6)), [2, 3]).flatten().shape), [6])

    def test_transpose(self):
        r = T([1,2,3,4,5,6], [2, 3]).transpose(0, 1)
        self.assertEqual(list(r.shape), [3, 2])
        self.assertTrue(allclose(r.storage(), [1,4,2,5,3,6]))

    def test_unsqueeze(self):
        t = T([1.0, 2.0, 3.0], [3])
        self.assertEqual(list(t.unsqueeze(0).shape), [1, 3])
        self.assertEqual(list(t.unsqueeze(1).shape), [3, 1])

    def test_reshape_backward(self):
        a = T(list(range(1, 7)), [2, 3], grad=True)
        blade.ops.sum(a.reshape([6])).backward()
        self.assertTrue(allclose(a.grad.storage(), [1.0] * 6))


# ===========================================================================
# 10. Autograd — multi-node chains
# ===========================================================================

class TestAutograd(unittest.TestCase):

    def test_two_node_chain(self):
        a = T([1.0, -2.0, 3.0], [3], grad=True)
        blade.ops.sum(blade.ops.relu(a)).backward()
        self.assertTrue(allclose(a.grad.storage(), [1, 0, 1]))

    def test_three_node_chain(self):
        a = T([0.0, 1.0], [2], grad=True)
        blade.ops.sum(blade.ops.exp(a * 2.0)).backward()
        self.assertTrue(allclose(a.grad.storage(), [2 * math.exp(2 * x) for x in [0, 1]]))

    def test_gradient_accumulates_across_uses(self):
        a = T([1.0, 2.0, 3.0], [3], grad=True)
        blade.ops.sum(a * a).backward()
        self.assertTrue(allclose(a.grad.storage(), [2, 4, 6]))

    def test_leaf_grad_zero_before_backward(self):
        a = T([1.0], [1], grad=True)
        self.assertTrue(allclose(a.grad.storage(), [0.0]))

    def test_zero_grad_clears(self):
        a = T([1.0, 2.0], [2], grad=True)
        blade.ops.sum(a).backward()
        a.zero_grad()
        self.assertTrue(allclose(a.grad.storage(), [0.0, 0.0]))


# ===========================================================================
# 11. nn.Linear
# ===========================================================================

class TestLinear(unittest.TestCase):

    def test_output_shape(self):
        self.assertEqual(list(nn.Linear(4, 8)(blade.Tensor.randn([3, 4])).shape), [3, 8])

    def test_parameters_count(self):
        self.assertEqual(len(nn.Linear(4, 8).parameters()), 2)  # weight + bias

    def test_no_bias(self):
        self.assertEqual(len(nn.Linear(4, 8, bias=False).parameters()), 1)

    def test_bias_grad_computed(self):
        layer = nn.Linear(3, 2)
        blade.ops.sum(layer(blade.Tensor.randn([4, 3]))).backward()
        self.assertTrue(all(p.grad is not None for p in layer.parameters()))

    def test_bias_shape(self):
        layer = nn.Linear(3, 5)
        self.assertEqual(list(layer.bias.shape), [5])

    def test_bias_adds_offset(self):
        # weight = zeros, bias = [1, 2] -> output should equal bias for any input
        layer = nn.Linear(3, 2, bias=True)
        w = layer.weight  # shape [2, 3]
        for i in range(2):
            for j in range(3):
                w.set([i, j], 0.0)
        layer.bias.set([0], 1.0)
        layer.bias.set([1], 2.0)
        x = T([5.0, 5.0, 5.0], [1, 3])
        out = layer(x)
        self.assertTrue(allclose(out.storage(), [1.0, 2.0]))


# ===========================================================================
# 12. nn.Flatten
# ===========================================================================

class TestFlatten(unittest.TestCase):

    def test_output_shape(self):
        self.assertEqual(list(nn.Flatten()(blade.Tensor.randn([4, 3, 2])).shape), [4, 6])

    def test_preserves_values(self):
        out = nn.Flatten()(T([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]))
        self.assertTrue(allclose(out.storage(), [1, 2, 3, 4, 5, 6]))


# ===========================================================================
# 13. mse_loss
# ===========================================================================

class TestMSELoss(unittest.TestCase):

    def test_zero_when_equal(self):
        a = T([1.0, 2.0, 3.0], [3])
        self.assertAlmostEqual(nn.mse_loss(a, a).item(), 0.0, places=5)

    def test_known_value(self):
        self.assertAlmostEqual(nn.mse_loss(T([0, 0], [2]), T([1, 3], [2])).item(), 5.0, places=5)

    def test_backward(self):
        pred = T([0.0, 0.0], [2], grad=True)
        nn.mse_loss(pred, T([1.0, 3.0], [2])).backward()
        self.assertTrue(allclose(pred.grad.storage(), [-1.0, -3.0]))

    def test_shape_mismatch_raises(self):
        with self.assertRaises(Exception):
            nn.mse_loss(T([1, 2], [2]), T([1, 2, 3], [3]))


# ===========================================================================
# 14. nll_loss
# ===========================================================================

class TestNLLLoss(unittest.TestCase):

    def test_single_sample(self):
        lp = T([math.log(0.1), math.log(0.7), math.log(0.2)], [1, 3])
        self.assertAlmostEqual(nn.nll_loss(lp, T([1.0], [1])).item(), -math.log(0.7), places=5)

    def test_two_samples(self):
        lp = T([math.log(0.1), math.log(0.7), math.log(0.2),
                math.log(0.3), math.log(0.3), math.log(0.4)], [2, 3])
        expected = -(math.log(0.7) + math.log(0.4)) / 2.0
        self.assertAlmostEqual(nn.nll_loss(lp, T([1.0, 2.0], [2])).item(), expected, places=5)

    def test_perfect_predictions_low_loss(self):
        lp = T([math.log(0.999), math.log(0.001),
                math.log(0.001), math.log(0.999)], [2, 2])
        self.assertLess(nn.nll_loss(lp, T([0.0, 1.0], [2])).item(), 0.01)

    def test_worst_predictions_high_loss(self):
        lp = T([math.log(0.001), math.log(0.999),
                math.log(0.999), math.log(0.001)], [2, 2])
        self.assertGreater(nn.nll_loss(lp, T([0.0, 1.0], [2])).item(), 4.0)

    def test_non_negative(self):
        lp = T([math.log(0.3), math.log(0.3), math.log(0.4),
                math.log(0.5), math.log(0.3), math.log(0.2)], [2, 3])
        self.assertGreaterEqual(nn.nll_loss(lp, T([0.0, 2.0], [2])).item(), 0.0)

    def test_backward(self):
        # Gradient = -1/N at target positions, 0 elsewhere.
        lp = T([math.log(0.2), math.log(0.5), math.log(0.3),
                math.log(0.6), math.log(0.2), math.log(0.2)], [2, 3], grad=True)
        nn.nll_loss(lp, T([1.0, 0.0], [2])).backward()
        g = lp.grad.storage()
        self.assertAlmostEqual(g[1], -0.5, places=5)   # (0, target=1)
        self.assertAlmostEqual(g[3], -0.5, places=5)   # (1, target=0)
        self.assertAlmostEqual(g[0],  0.0, places=5)
        self.assertAlmostEqual(g[2],  0.0, places=5)
        self.assertAlmostEqual(g[4],  0.0, places=5)
        self.assertAlmostEqual(g[5],  0.0, places=5)


# ===========================================================================
# 15. cross_entropy
# ===========================================================================

class TestCrossEntropy(unittest.TestCase):

    def _ref(self, logits_rows, targets):
        total = sum(ref_log_softmax(row)[int(t)] for row, t in zip(logits_rows, targets))
        return -total / len(targets)

    def test_known_value(self):
        inp = T([1,2,3, 1,2,3], [2, 3])
        tgt = T([2.0, 0.0], [2])
        self.assertAlmostEqual(nn.cross_entropy(inp, tgt).item(), self._ref([[1,2,3],[1,2,3]], [2,0]), places=5)

    def test_perfect_logits_low_loss(self):
        inp = T([10, 0, 0, 0, 10, 0], [2, 3])
        self.assertLess(nn.cross_entropy(inp, T([0.0, 1.0], [2])).item(), 0.01)

    def test_uniform_logits_equals_log_C(self):
        inp = T([0.0] * 8, [2, 4])
        self.assertAlmostEqual(nn.cross_entropy(inp, T([0.0, 1.0], [2])).item(), math.log(4), places=5)

    def test_non_negative(self):
        inp = T([1.0, 2.0, 0.5, 0.5, 1.5, 2.5], [2, 3])
        self.assertGreaterEqual(nn.cross_entropy(inp, T([0.0, 2.0], [2])).item(), 0.0)

    def test_backward_reaches_all_logits(self):
        inp = T([1,2,3, 4,5,6], [2, 3], grad=True)
        nn.cross_entropy(inp, T([0.0, 2.0], [2])).backward()
        self.assertEqual(len(inp.grad.storage()), 6)
        self.assertFalse(allclose(inp.grad.storage(), [0.0] * 6))

    def test_module_wrapper(self):
        inp = T([1,2,3, 3,2,1], [2, 3])
        tgt = T([2.0, 0.0], [2])
        self.assertAlmostEqual(nn.cross_entropy(inp, tgt).item(),
                               nn.CrossEntropyLoss()(inp, tgt).item(), places=5)


# ===========================================================================
# 16. SGD optimizer
# ===========================================================================

class TestSGD(unittest.TestCase):

    def test_single_step(self):
        p = T([1.0], [1], grad=True)
        opt = optim.SGD([p], lr=0.1)
        blade.ops.sum(p * p).backward()
        opt.step()
        self.assertAlmostEqual(p.storage()[0], 0.8, places=5)

    def test_zero_grad(self):
        p = T([1.0], [1], grad=True)
        opt = optim.SGD([p], lr=0.1)
        blade.ops.sum(p * p).backward()
        opt.zero_grad()
        self.assertTrue(allclose(p.grad.storage(), [0.0]))

    def test_descends_loss(self):
        p = T([3.0], [1], grad=True)
        opt = optim.SGD([p], lr=0.01)
        losses = []
        for _ in range(20):
            opt.zero_grad()
            loss = blade.ops.sum(p * p)
            losses.append(loss.item())
            loss.backward()
            opt.step()
        self.assertLess(losses[-1], losses[0])


# ===========================================================================
# 17. Adam optimizer
# ===========================================================================

class TestAdam(unittest.TestCase):

    def test_single_step_moves_param(self):
        p = T([1.0], [1], grad=True)
        opt = optim.Adam([p], lr=0.01)
        blade.ops.sum(p * p).backward()
        before = p.storage()[0]
        opt.step()
        self.assertLess(p.storage()[0], before)

    def test_descends_loss(self):
        p = T([3.0], [1], grad=True)
        opt = optim.Adam([p], lr=0.1)
        losses = []
        for _ in range(30):
            opt.zero_grad()
            loss = blade.ops.sum(p * p)
            losses.append(loss.item())
            loss.backward()
            opt.step()
        self.assertLess(losses[-1], losses[0])

    def test_zero_grad(self):
        p = T([1.0], [1], grad=True)
        opt = optim.Adam([p], lr=0.01)
        blade.ops.sum(p).backward()
        opt.zero_grad()
        self.assertTrue(allclose(p.grad.storage(), [0.0]))


# ===========================================================================
# 18. DataLoader.collate  (requires PyDataset trampoline in bindings)
# ===========================================================================

class TestDataLoaderCollate(unittest.TestCase):

    def _make_loader(self, n_samples, input_shape, batch_size, shuffle=False):
        import math as _math

        class SimpleDataset(data.Dataset):
            def size(self):
                return n_samples
            def get(self, idx):
                x = blade.Tensor.from_data(input_shape,
                    [float(idx)] * int(_math.prod(input_shape)))
                y = blade.Tensor.from_data([1], [float(idx)])
                return (x, y)

        return data.DataLoader(SimpleDataset(), batch_size, shuffle)

    def test_batch_input_shape(self):
        x, _ = next(iter(self._make_loader(4, [3], batch_size=4)))
        self.assertEqual(list(x.shape), [4, 3])

    def test_batch_label_shape(self):
        # collate squeezes scalar labels into (N,)
        _, y = next(iter(self._make_loader(4, [3], batch_size=4)))
        self.assertEqual(list(y.shape), [4])

    def test_partial_batch_at_end(self):
        batches = list(self._make_loader(5, [2], batch_size=4))
        self.assertEqual(list(batches[0][0].shape), [4, 2])
        self.assertEqual(list(batches[1][0].shape), [1, 2])

    def test_correct_values_in_batch(self):
        x, y = next(iter(self._make_loader(3, [3], batch_size=3)))
        self.assertTrue(allclose(x.storage(), [0,0,0, 1,1,1, 2,2,2]))
        self.assertTrue(allclose(y.storage(), [0.0, 1.0, 2.0]))

    def test_num_batches(self):
        self.assertEqual(len(self._make_loader(10, [4], batch_size=3)), 4)

    def test_2d_input_stacking(self):
        x, _ = next(iter(self._make_loader(4, [2, 3], batch_size=4)))
        self.assertEqual(list(x.shape), [4, 2, 3])


# ===========================================================================
# 19. MLP integration test (no data files needed)
# ===========================================================================

class TestMLPIntegration(unittest.TestCase):

    def test_mlp_loss_decreases(self):
        """Train a tiny MLP on a trivial dataset for 30 steps; loss must fall."""
        class TinyMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 8)
                self.fc2 = nn.Linear(8, 3)
                self.relu = nn.ReLU()
            def forward(self, x):
                return self.fc2(self.relu(self.fc1(x)))

        model = TinyMLP()
        opt   = optim.Adam(model.parameters(), lr=1e-2)
        # 8 samples, 4 features, 3 classes
        X = blade.Tensor.randn([8, 4])
        y = T([0,1,2,0,1,2,0,1], [8])

        losses = []
        for _ in range(30):
            opt.zero_grad()
            loss = nn.cross_entropy(model(X), y)
            losses.append(loss.item())
            loss.backward()
            opt.step()

        self.assertLess(losses[-1], losses[0],
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}")

    def test_module_parameters_count(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(3, 4)
                self.fc2 = nn.Linear(4, 2)
            def forward(self, x):
                return self.fc2(self.fc1(x))
        net = Net()
        # fc1: weight(4x3) + bias(4) = 2 tensors
        # fc2: weight(2x4) + bias(2) = 2 tensors
        self.assertEqual(len(net.parameters()), 4)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
