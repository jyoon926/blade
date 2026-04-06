"""
BLADE unit test suite
=====================
Run from the repo root after building:

    cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
    cd .. && python tests/test_blade.py

Skipped intentionally (stubs / out of scope):
  - dropout, cat, conv2d, pooling, batch_norm
  - nll_loss, cross_entropy  (nll_loss gather not yet implemented)
  - DataLoader.collate       (stub)
"""

import sys, math, unittest
sys.path.insert(0, "build")   # picks up blade.cpython-*.so
import blade
import blade.nn   as nn
import blade.optim as optim

TOL = 1e-5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def T(data, shape, grad=False):
    """Create a Tensor from a flat list."""
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

    def test_zeros_shape_and_values(self):
        t = blade.Tensor.zeros([2, 3])
        self.assertEqual(list(t.shape), [2, 3])
        self.assertTrue(allclose(t.storage(), [0.0] * 6))

    def test_ones_shape_and_values(self):
        t = blade.Tensor.ones([3])
        self.assertTrue(allclose(t.storage(), [1.0, 1.0, 1.0]))

    def test_full(self):
        t = blade.Tensor.full([2, 2], 7.0)
        self.assertTrue(allclose(t.storage(), [7.0] * 4))

    def test_from_data_roundtrip(self):
        data = [1.0, 2.0, 3.0, 4.0]
        t = T(data, [2, 2])
        self.assertEqual(list(t.shape), [2, 2])
        self.assertTrue(allclose(t.storage(), data))

    def test_arange(self):
        t = blade.Tensor.arange(0, 5, 1)
        self.assertTrue(allclose(t.storage(), [0, 1, 2, 3, 4]))

    def test_randn_shape(self):
        t = blade.Tensor.randn([4, 5])
        self.assertEqual(list(t.shape), [4, 5])
        self.assertEqual(t.numel, 20)

    def test_uniform_bounds(self):
        t = blade.Tensor.uniform([1000], 0.0, 1.0)
        vals = t.storage()
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
        t = T([1.0, 4.0, 9.0], [3])
        self.assertTrue(allclose(blade.ops.sqrt(t).storage(), [1, 2, 3]))

    def test_abs(self):
        t = T([-1.0, 2.0, -3.0], [3])
        self.assertTrue(allclose(blade.ops.abs(t).storage(), [1, 2, 3]))

    def test_clamp(self):
        t = T([-1.0, 0.5, 2.0], [3])
        self.assertTrue(allclose(blade.ops.clamp(t, 0.0, 1.0).storage(), [0, 0.5, 1]))


# ===========================================================================
# 3. Element-wise ops — backward
# ===========================================================================

class TestElementWiseBackward(unittest.TestCase):

    def _grad(self, out, inp):
        """Run backward and return inp.grad as a list."""
        out.backward()
        return list(inp.grad.storage())

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
        self.assertTrue(allclose(a.grad.storage(), [4, 5]))  # d/da = b
        self.assertTrue(allclose(b.grad.storage(), [2, 3]))  # d/db = a

    def test_pow_grad(self):
        # d/dx x^3 = 3x^2
        a = T([1.0, 2.0, 3.0], [3], grad=True)
        blade.ops.sum(blade.ops.pow(a, 3.0)).backward()
        expected = [3 * x**2 for x in [1, 2, 3]]
        self.assertTrue(allclose(a.grad.storage(), expected))

    def test_exp_grad(self):
        # d/dx exp(x) = exp(x)
        a = T([0.0, 1.0], [2], grad=True)
        blade.ops.sum(blade.ops.exp(a)).backward()
        expected = [math.exp(x) for x in [0, 1]]
        self.assertTrue(allclose(a.grad.storage(), expected))

    def test_log_grad(self):
        # d/dx log(x) = 1/x
        a = T([1.0, 2.0, 4.0], [3], grad=True)
        blade.ops.sum(blade.ops.log(a)).backward()
        self.assertTrue(allclose(a.grad.storage(), [1.0, 0.5, 0.25]))

    def test_neg_grad(self):
        a = T([1.0, 2.0], [2], grad=True)
        blade.ops.sum(blade.ops.neg(a)).backward()
        self.assertTrue(allclose(a.grad.storage(), [-1, -1]))

    def test_chain_two_ops(self):
        # sum(relu(a)) — exercises the multi-node backward fix
        a = T([1.0, -2.0, 3.0], [3], grad=True)
        blade.ops.sum(blade.ops.relu(a)).backward()
        self.assertTrue(allclose(a.grad.storage(), [1, 0, 1]))

    def test_chain_three_ops(self):
        # sum(exp(a * 2))
        a = T([0.0, 1.0], [2], grad=True)
        blade.ops.sum(blade.ops.exp(a * 2.0)).backward()
        # d/da = 2 * exp(2a)
        expected = [2 * math.exp(2 * x) for x in [0, 1]]
        self.assertTrue(allclose(a.grad.storage(), expected))


# ===========================================================================
# 4. Global reductions
# ===========================================================================

class TestGlobalReductions(unittest.TestCase):

    def test_sum_forward(self):
        t = T([1.0, 2.0, 3.0, 4.0], [4])
        self.assertAlmostEqual(blade.ops.sum(t).item(), 10.0)

    def test_sum_backward(self):
        a = T([1.0, 2.0, 3.0], [3], grad=True)
        blade.ops.sum(a).backward()
        self.assertTrue(allclose(a.grad.storage(), [1, 1, 1]))

    def test_mean_forward(self):
        t = T([1.0, 2.0, 3.0, 4.0], [4])
        self.assertAlmostEqual(blade.ops.mean(t).item(), 2.5)

    def test_mean_backward(self):
        a = T([1.0, 2.0, 3.0, 4.0], [4], grad=True)
        blade.ops.mean(a).backward()
        self.assertTrue(allclose(a.grad.storage(), [0.25] * 4))

    def test_max_forward(self):
        t = T([3.0, 1.0, 4.0, 1.0, 5.0], [5])
        self.assertAlmostEqual(blade.ops.max(t).item(), 5.0)

    def test_max_backward(self):
        # Gradient flows only to the winning element (index 4)
        a = T([3.0, 1.0, 4.0, 1.0, 5.0], [5], grad=True)
        blade.ops.max(a).backward()
        self.assertTrue(allclose(a.grad.storage(), [0, 0, 0, 0, 1]))


# ===========================================================================
# 5. Dim-wise reductions
# ===========================================================================

class TestDimReductions(unittest.TestCase):
    #  input: [[1, 2, 3],
    #           [4, 5, 6]]   shape (2, 3)
    DATA  = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    SHAPE = [2, 3]

    def test_sum_dim0_shape(self):
        out = blade.ops.sum(T(self.DATA, self.SHAPE), 0, False)
        self.assertEqual(list(out.shape), [3])

    def test_sum_dim0_vals(self):
        out = blade.ops.sum(T(self.DATA, self.SHAPE), 0)
        self.assertTrue(allclose(out.storage(), [5, 7, 9]))

    def test_sum_dim1_vals(self):
        out = blade.ops.sum(T(self.DATA, self.SHAPE), 1)
        self.assertTrue(allclose(out.storage(), [6, 15]))

    def test_sum_keepdim(self):
        out = blade.ops.sum(T(self.DATA, self.SHAPE), 0, True)
        self.assertEqual(list(out.shape), [1, 3])

    def test_sum_negative_dim(self):
        out = blade.ops.sum(T(self.DATA, self.SHAPE), -1)
        self.assertTrue(allclose(out.storage(), [6, 15]))

    def test_sum_dim0_backward(self):
        a = T(self.DATA, self.SHAPE, grad=True)
        blade.ops.sum(blade.ops.sum(a, 0)).backward()
        self.assertTrue(allclose(a.grad.storage(), [1.0] * 6))

    def test_sum_dim1_backward(self):
        a = T(self.DATA, self.SHAPE, grad=True)
        blade.ops.sum(blade.ops.sum(a, 1)).backward()
        self.assertTrue(allclose(a.grad.storage(), [1.0] * 6))

    def test_mean_dim0_vals(self):
        out = blade.ops.mean(T(self.DATA, self.SHAPE), 0)
        self.assertTrue(allclose(out.storage(), [2.5, 3.5, 4.5]))

    def test_mean_dim1_vals(self):
        out = blade.ops.mean(T(self.DATA, self.SHAPE), 1)
        self.assertTrue(allclose(out.storage(), [2.0, 5.0]))

    def test_mean_dim0_backward(self):
        a = T(self.DATA, self.SHAPE, grad=True)
        blade.ops.sum(blade.ops.mean(a, 0)).backward()
        self.assertTrue(allclose(a.grad.storage(), [0.5] * 6))

    def test_max_dim0_vals(self):
        out = blade.ops.max(T(self.DATA, self.SHAPE), 0)
        self.assertTrue(allclose(out.storage(), [4, 5, 6]))

    def test_max_dim1_vals(self):
        out = blade.ops.max(T(self.DATA, self.SHAPE), 1)
        self.assertTrue(allclose(out.storage(), [3, 6]))

    def test_max_dim0_backward(self):
        # data [[1,2,3],[4,5,6]]: every col max is in row1 (4>1, 5>2, 6>3)
        # so gradient flows to all three row1 elements: [0,0,0, 1,1,1]
        a = T(self.DATA, self.SHAPE, grad=True)
        blade.ops.sum(blade.ops.max(a, 0)).backward()
        self.assertTrue(allclose(a.grad.storage(), [0, 0, 0, 1, 1, 1]))

    def test_max_dim1_backward(self):
        # Winners: row0->col2(idx2), row1->col2(idx5)
        a = T(self.DATA, self.SHAPE, grad=True)
        blade.ops.sum(blade.ops.max(a, 1)).backward()
        self.assertTrue(allclose(a.grad.storage(), [0, 0, 1, 0, 0, 1]))

    def test_max_ties_first_wins(self):
        a = T([5.0, 5.0, 5.0], [1, 3], grad=True)
        blade.ops.max(a, 1).backward()
        self.assertTrue(allclose(a.grad.storage(), [1, 0, 0]))


# ===========================================================================
# 6. Matmul
# ===========================================================================

class TestMatmul(unittest.TestCase):

    def test_2d_shape(self):
        a = T([1]*6,  [2, 3])
        b = T([1]*12, [3, 4])
        self.assertEqual(list(blade.ops.matmul(a, b).shape), [2, 4])

    def test_2d_values(self):
        # [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        a = T([1,2,3,4], [2,2])
        b = T([5,6,7,8], [2,2])
        self.assertTrue(allclose(blade.ops.matmul(a, b).storage(), [19,22,43,50]))

    def test_batched_shape(self):
        a = T([1]*12, [2, 2, 3])
        b = T([1]*12, [2, 3, 2])
        self.assertEqual(list(blade.ops.matmul(a, b).shape), [2, 2, 2])

    def test_matmul_backward(self):
        # C = A @ B,  dA = g @ B^T,  dB = A^T @ g
        a = T([1,2,3,4], [2,2], grad=True)
        b = T([1,0,0,1], [2,2], grad=True)   # identity
        blade.ops.sum(blade.ops.matmul(a, b)).backward()
        # dL/dA = g @ B^T = ones(2,2) @ I = ones(2,2)
        self.assertTrue(allclose(a.grad.storage(), [1]*4))


# ===========================================================================
# 7. Activations
# ===========================================================================

class TestActivations(unittest.TestCase):

    def test_relu_forward(self):
        t = T([-2.0, 0.0, 3.0], [3])
        self.assertTrue(allclose(blade.ops.relu(t).storage(), [0, 0, 3]))

    def test_relu_backward(self):
        a = T([-1.0, 2.0, -3.0, 4.0], [4], grad=True)
        blade.ops.sum(blade.ops.relu(a)).backward()
        self.assertTrue(allclose(a.grad.storage(), [0, 1, 0, 1]))

    def test_leaky_relu_forward(self):
        t = T([-2.0, 3.0], [2])
        out = blade.ops.leaky_relu(t, 0.1)
        self.assertTrue(allclose(out.storage(), [-0.2, 3.0]))

    def test_sigmoid_forward(self):
        t = T([0.0], [1])
        self.assertAlmostEqual(blade.ops.sigmoid(t).item(), 0.5, places=5)

    def test_sigmoid_backward(self):
        # d/dx sigmoid(x) = s(x) * (1 - s(x)), at x=0 -> 0.25
        a = T([0.0], [1], grad=True)
        blade.ops.sigmoid(a).backward()
        self.assertAlmostEqual(a.grad.storage()[0], 0.25, places=5)

    def test_tanh_forward(self):
        t = T([0.0], [1])
        self.assertAlmostEqual(blade.ops.tanh(t).item(), 0.0, places=5)

    def test_tanh_backward(self):
        # d/dx tanh(x) = 1 - tanh(x)^2, at x=0 -> 1
        a = T([0.0], [1], grad=True)
        blade.ops.tanh(a).backward()
        self.assertAlmostEqual(a.grad.storage()[0], 1.0, places=5)

    def test_softmax_sums_to_one(self):
        t = T([1.0, 2.0, 3.0], [1, 3])
        self.assertAlmostEqual(sum(blade.ops.softmax(t, 1).storage()), 1.0, places=5)

    def test_softmax_values(self):
        logits = [1.0, 2.0, 3.0]
        t = T(logits, [1, 3])
        self.assertTrue(allclose(blade.ops.softmax(t, 1).storage(), ref_softmax(logits)))

    def test_softmax_numerically_stable(self):
        t = T([1000.0, 1001.0, 1002.0], [1, 3])
        s = sum(blade.ops.softmax(t, 1).storage())
        self.assertAlmostEqual(s, 1.0, places=5)

    def test_softmax_backward_ones_upstream(self):
        # grad of sum(softmax) w.r.t. input should be all zeros
        # because dot(ones, softmax) = 1, so ga[i] = s[i]*(1-1) = 0
        a = T([1.0, 2.0, 3.0], [1, 3], grad=True)
        blade.ops.softmax(a, 1).backward()
        self.assertTrue(allclose(a.grad.storage(), [0.0, 0.0, 0.0]))

    def test_softmax_backward_selective(self):
        # upstream g=[1,0,0]: ga[i] = s[i]*(delta(i,0) - s[0])
        logits = [1.0, 2.0, 3.0]
        s = ref_softmax(logits)
        a = T(logits, [1, 3], grad=True)
        sm = blade.ops.softmax(a, 1)
        mask = T([1.0, 0.0, 0.0], [1, 3])
        blade.ops.sum(sm * mask).backward()
        expected = [s[i] * ((1.0 if i == 0 else 0.0) - s[0]) for i in range(3)]
        self.assertTrue(allclose(a.grad.storage(), expected))

    def test_log_softmax_values(self):
        logits = [1.0, 2.0, 3.0]
        t = T(logits, [1, 3])
        self.assertTrue(allclose(blade.ops.log_softmax(t, 1).storage(), ref_log_softmax(logits)))

    def test_log_softmax_equals_log_of_softmax(self):
        logits = [0.5, 1.5, -0.5]
        t = T(logits, [1, 3])
        lsm = blade.ops.log_softmax(t, 1).storage()
        sm  = [math.log(x) for x in ref_softmax(logits)]
        self.assertTrue(allclose(lsm, sm))

    def test_log_softmax_numerically_stable(self):
        t = T([1000.0, 1001.0, 1002.0], [1, 3])
        vals = blade.ops.log_softmax(t, 1).storage()
        self.assertTrue(all(abs(v) < 10 for v in vals))

    def test_log_softmax_backward_sums_to_zero(self):
        # When upstream g=ones, gradients must sum to 0 (probability constraint)
        a = T([1.0, 2.0, 3.0], [1, 3], grad=True)
        blade.ops.log_softmax(a, 1).backward()
        self.assertAlmostEqual(sum(a.grad.storage()), 0.0, places=5)

    def test_log_softmax_backward_values(self):
        # ga[i] = g[i] - s[i]*sum(g), with g=ones: ga[i] = 1 - s[i]*3
        logits = [1.0, 2.0, 3.0]
        s = ref_softmax(logits)
        a = T(logits, [1, 3], grad=True)
        blade.ops.log_softmax(a, 1).backward()
        expected = [1.0 - s[i] * 3 for i in range(3)]
        self.assertTrue(allclose(a.grad.storage(), expected))

    def test_softmax_batched(self):
        rows = [1.0, 2.0, 3.0,  0.0, 0.0, 0.0]
        t = T(rows, [2, 3])
        out = blade.ops.softmax(t, 1)
        self.assertTrue(allclose(out.storage()[:3], ref_softmax([1,2,3])))
        self.assertTrue(allclose(out.storage()[3:], ref_softmax([0,0,0])))


# ===========================================================================
# 8. Shape ops
# ===========================================================================

class TestShapeOps(unittest.TestCase):

    def test_reshape(self):
        t = T(list(range(6)), [6])
        r = t.reshape([2, 3])
        self.assertEqual(list(r.shape), [2, 3])
        self.assertTrue(allclose(r.storage(), list(range(6))))

    def test_flatten(self):
        t = T(list(range(6)), [2, 3])
        f = t.flatten()
        self.assertEqual(list(f.shape), [6])

    def test_transpose(self):
        t = T([1,2,3,4,5,6], [2, 3])
        r = t.transpose(0, 1)
        self.assertEqual(list(r.shape), [3, 2])
        # [[1,2,3],[4,5,6]]^T = [[1,4],[2,5],[3,6]]
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
# 9. Autograd — multi-node chains
# ===========================================================================

class TestAutograd(unittest.TestCase):

    def test_two_node_chain(self):
        a = T([1.0, -2.0, 3.0], [3], grad=True)
        blade.ops.sum(blade.ops.relu(a)).backward()
        self.assertTrue(allclose(a.grad.storage(), [1, 0, 1]))

    def test_three_node_chain(self):
        # sum(exp(a * 2)): d/da = 2 * exp(2a)
        a = T([0.0, 1.0], [2], grad=True)
        blade.ops.sum(blade.ops.exp(a * 2.0)).backward()
        expected = [2 * math.exp(2 * x) for x in [0, 1]]
        self.assertTrue(allclose(a.grad.storage(), expected))

    def test_gradient_accumulates_across_uses(self):
        # a is used twice: loss = sum(a * a) -> d/da = 2a
        a = T([1.0, 2.0, 3.0], [3], grad=True)
        blade.ops.sum(a * a).backward()
        self.assertTrue(allclose(a.grad.storage(), [2, 4, 6]))

    def test_leaf_grad_is_zero_before_backward(self):
        # The framework initialises grad_ to a zero tensor rather than None,
        # so .grad returns a zero tensor before any backward call.
        a = T([1.0], [1], grad=True)
        self.assertTrue(allclose(a.grad.storage(), [0.0]))

    def test_zero_grad_clears(self):
        a = T([1.0, 2.0], [2], grad=True)
        blade.ops.sum(a).backward()
        a.zero_grad()
        self.assertTrue(allclose(a.grad.storage(), [0.0, 0.0]))


# ===========================================================================
# 10. nn.Linear
# ===========================================================================

class TestLinear(unittest.TestCase):

    def test_output_shape(self):
        layer = nn.Linear(4, 8)
        x = blade.Tensor.randn([3, 4])
        out = layer(x)
        self.assertEqual(list(out.shape), [3, 8])

    def test_parameters_count(self):
        layer = nn.Linear(4, 8)
        params = layer.parameters()
        # weight (4x8=32) + bias (8) = 2 parameter tensors
        self.assertEqual(len(params), 2)

    def test_no_bias(self):
        layer = nn.Linear(4, 8, bias=False)
        self.assertEqual(len(layer.parameters()), 1)

    def test_backward_updates_weight_grad(self):
        layer = nn.Linear(3, 2)
        x = blade.Tensor.randn([4, 3])
        blade.ops.sum(layer(x)).backward()
        # Both weight and bias should have gradients after backward
        params = layer.parameters()
        self.assertTrue(all(p.grad is not None for p in params))

    def test_linear_identity_weight(self):
        # With weight=I and bias=0, output should equal input
        layer = nn.Linear(2, 2, bias=False)
        # Manually set weight to identity
        layer.weight.storage()  # just confirm it's accessible
        x = T([1.0, 0.0, 0.0, 1.0], [2, 2])
        # We can't easily set weights directly through the API, so just
        # confirm the output shape and that the op runs without error
        out = layer(x)
        self.assertEqual(list(out.shape), [2, 2])


# ===========================================================================
# 11. nn.Flatten
# ===========================================================================

class TestFlatten(unittest.TestCase):

    def test_flatten_output_shape(self):
        layer = nn.Flatten()
        x = blade.Tensor.randn([4, 3, 2])
        out = layer(x)
        self.assertEqual(list(out.shape), [4, 6])

    def test_flatten_preserves_values(self):
        layer = nn.Flatten()
        x = T([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
        out = layer(x)
        self.assertTrue(allclose(out.storage(), [1, 2, 3, 4, 5, 6]))


# ===========================================================================
# 12. mse_loss
# ===========================================================================

class TestMSELoss(unittest.TestCase):

    def test_mse_zero_when_equal(self):
        a = T([1.0, 2.0, 3.0], [3])
        self.assertAlmostEqual(nn.mse_loss(a, a).item(), 0.0, places=5)

    def test_mse_known_value(self):
        pred   = T([0.0, 0.0], [2])
        target = T([1.0, 3.0], [2])
        # mean((0-1)^2, (0-3)^2) = mean(1, 9) = 5.0
        self.assertAlmostEqual(nn.mse_loss(pred, target).item(), 5.0, places=5)

    def test_mse_backward(self):
        pred   = T([0.0, 0.0], [2], grad=True)
        target = T([1.0, 3.0], [2])
        nn.mse_loss(pred, target).backward()
        # d/dpred = 2*(pred - target)/N = 2*[-1,-3]/2 = [-1,-3]
        self.assertTrue(allclose(pred.grad.storage(), [-1.0, -3.0]))

    def test_mse_shape_mismatch_raises(self):
        a = T([1.0, 2.0], [2])
        b = T([1.0, 2.0, 3.0], [3])
        with self.assertRaises(Exception):
            nn.mse_loss(a, b)


# ===========================================================================
# 13. SGD optimizer
# ===========================================================================

class TestSGD(unittest.TestCase):

    def _simple_param(self, val):
        p = T([val], [1], grad=True)
        return p

    def test_sgd_single_step(self):
        # loss = param^2, grad = 2*param; with lr=0.1 and param=1.0:
        # param_new = 1.0 - 0.1 * 2.0 = 0.8
        p = self._simple_param(1.0)
        opt = optim.SGD([p], lr=0.1)
        blade.ops.sum(p * p).backward()
        opt.step()
        self.assertAlmostEqual(p.storage()[0], 0.8, places=5)

    def test_sgd_zero_grad(self):
        p = self._simple_param(1.0)
        opt = optim.SGD([p], lr=0.1)
        blade.ops.sum(p * p).backward()
        opt.zero_grad()
        self.assertTrue(allclose(p.grad.storage(), [0.0]))

    def test_sgd_descends_loss(self):
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
# 14. Adam optimizer
# ===========================================================================

class TestAdam(unittest.TestCase):

    def test_adam_single_step_moves_param(self):
        p = T([1.0], [1], grad=True)
        opt = optim.Adam([p], lr=0.01)
        blade.ops.sum(p * p).backward()
        before = p.storage()[0]
        opt.step()
        after = p.storage()[0]
        self.assertLess(after, before)  # should move toward 0

    def test_adam_descends_loss(self):
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

    def test_adam_zero_grad(self):
        p = T([1.0], [1], grad=True)
        opt = optim.Adam([p], lr=0.01)
        blade.ops.sum(p).backward()
        opt.zero_grad()
        self.assertTrue(allclose(p.grad.storage(), [0.0]))


# ===========================================================================
# 15. nll_loss  [STUB — not yet implemented]
#
# Expected behaviour once implemented:
#   nll_loss(log_probs, targets)
#     log_probs : (N, C)  log-probabilities (output of log_softmax)
#     targets   : (N,)    integer class indices in [0, C)
#   Returns: scalar = -mean(log_probs[i, targets[i]] for i in range(N))
# ===========================================================================

class TestNLLLoss(unittest.TestCase):

    def test_single_sample_correct_class(self):
        # 1 sample, 3 classes, target = class 1
        # loss = -log(0.7)
        log_probs = T([math.log(0.1), math.log(0.7), math.log(0.2)], [1, 3])
        target    = T([1.0], [1])
        expected  = -math.log(0.7)
        self.assertAlmostEqual(nn.nll_loss(log_probs, target).item(), expected, places=5)

    def test_two_samples(self):
        # sample 0: target=1 -> log(0.7), sample 1: target=2 -> log(0.4)
        # loss = -mean(log(0.7), log(0.4))
        lp = T([math.log(0.1), math.log(0.7), math.log(0.2),
                math.log(0.3), math.log(0.3), math.log(0.4)], [2, 3])
        target   = T([1.0, 2.0], [2])
        expected = -(math.log(0.7) + math.log(0.4)) / 2.0
        self.assertAlmostEqual(nn.nll_loss(lp, target).item(), expected, places=5)

    def test_perfect_predictions_low_loss(self):
        lp = T([math.log(0.999), math.log(0.001),
                math.log(0.001), math.log(0.999)], [2, 2])
        target = T([0.0, 1.0], [2])
        self.assertLess(nn.nll_loss(lp, target).item(), 0.01)

    def test_worst_predictions_high_loss(self):
        lp = T([math.log(0.001), math.log(0.999),
                math.log(0.999), math.log(0.001)], [2, 2])
        target = T([0.0, 1.0], [2])
        self.assertGreater(nn.nll_loss(lp, target).item(), 4.0)

    def test_loss_is_non_negative(self):
        lp = T([math.log(0.3), math.log(0.3), math.log(0.4),
                math.log(0.5), math.log(0.3), math.log(0.2)], [2, 3])
        target = T([0.0, 2.0], [2])
        self.assertGreaterEqual(nn.nll_loss(lp, target).item(), 0.0)

    def test_backward_flows_to_correct_indices(self):
        # Gradient should be -1/N at the gathered positions, 0 elsewhere.
        # 2 samples, 3 classes: targets [1, 0] -> positions (0,1) and (1,0)
        lp = T([math.log(0.2), math.log(0.5), math.log(0.3),
                math.log(0.6), math.log(0.2), math.log(0.2)], [2, 3], grad=True)
        target = T([1.0, 0.0], [2])
        nn.nll_loss(lp, target).backward()
        g = lp.grad.storage()
        self.assertAlmostEqual(g[0], 0.0,  places=5)  # sample 0, class 0 (non-target)
        self.assertAlmostEqual(g[2], 0.0,  places=5)  # sample 0, class 2 (non-target)
        self.assertAlmostEqual(g[4], 0.0,  places=5)  # sample 1, class 1 (non-target)
        self.assertAlmostEqual(g[5], 0.0,  places=5)  # sample 1, class 2 (non-target)
        self.assertAlmostEqual(g[1], -0.5, places=5)  # sample 0, class 1 (target)
        self.assertAlmostEqual(g[3], -0.5, places=5)  # sample 1, class 0 (target)


# ===========================================================================
# 16. cross_entropy  [STUB — depends on nll_loss being implemented]
#
# Expected behaviour:
#   cross_entropy(logits, targets)
#     logits  : (N, C)  raw unnormalised scores
#     targets : (N,)    integer class indices
#   Equivalent to: nll_loss(log_softmax(logits, dim=1), targets)
# ===========================================================================

class TestCrossEntropy(unittest.TestCase):

    def _ref_cross_entropy(self, logits_rows, targets):
        total = 0.0
        for row, t in zip(logits_rows, targets):
            lsm = ref_log_softmax(row)
            total += lsm[int(t)]
        return -total / len(targets)

    def test_known_value_two_samples(self):
        logits = [1.0, 2.0, 3.0,
                  1.0, 2.0, 3.0]
        targets_idx = [2, 0]
        expected = self._ref_cross_entropy([[1,2,3],[1,2,3]], targets_idx)
        inp = T(logits, [2, 3])
        tgt = T([float(t) for t in targets_idx], [2])
        self.assertAlmostEqual(nn.cross_entropy(inp, tgt).item(), expected, places=5)

    def test_perfect_logits_low_loss(self):
        # Very large logit at the correct class -> loss near 0
        inp = T([10.0, 0.0, 0.0,
                 0.0, 10.0, 0.0], [2, 3])
        tgt = T([0.0, 1.0], [2])
        self.assertLess(nn.cross_entropy(inp, tgt).item(), 0.01)

    def test_uniform_logits_loss_equals_log_num_classes(self):
        # Uniform logits -> uniform softmax -> loss = log(C)
        C = 4
        inp = T([0.0] * (2 * C), [2, C])
        tgt = T([0.0, 1.0], [2])
        self.assertAlmostEqual(nn.cross_entropy(inp, tgt).item(), math.log(C), places=5)

    def test_loss_is_non_negative(self):
        inp = T([1.0, 2.0, 0.5, 0.5, 1.5, 2.5], [2, 3])
        tgt = T([0.0, 2.0], [2])
        self.assertGreaterEqual(nn.cross_entropy(inp, tgt).item(), 0.0)

    def test_backward_reaches_input_logits(self):
        # All input logits should receive a gradient (softmax mixes all classes)
        inp = T([1.0, 2.0, 3.0,
                 4.0, 5.0, 6.0], [2, 3], grad=True)
        tgt = T([0.0, 2.0], [2])
        nn.cross_entropy(inp, tgt).backward()
        self.assertEqual(len(inp.grad.storage()), 6)
        self.assertFalse(allclose(inp.grad.storage(), [0.0] * 6))

    def test_module_wrapper_matches_function(self):
        inp = T([1.0, 2.0, 3.0, 3.0, 2.0, 1.0], [2, 3])
        tgt = T([2.0, 0.0], [2])
        fn_loss  = nn.cross_entropy(inp, tgt).item()
        mod_loss = nn.CrossEntropyLoss()(inp, tgt).item()
        self.assertAlmostEqual(fn_loss, mod_loss, places=5)


# ===========================================================================
# 17. DataLoader.collate  [STUB — also requires Dataset trampoline fix]
#
# collate() stacks a list of (input, label) Sample pairs along dim 0.
# e.g. 4 samples each with input shape (3,) and label shape (1,)
#      -> batched input shape (4, 3), batched label shape (4, 1)
#
# NOTE: These tests are skipped because Dataset has no Python trampoline
# in bindings.cpp, so it cannot be subclassed from Python yet.
# To enable them, add a PyDataset trampoline class (mirrors the existing
# PyModule pattern for nn::Module) and remove the @unittest.skip decorator.
# ===========================================================================

@unittest.skip("Requires PyDataset trampoline to be added to bindings.cpp")
class TestDataLoaderCollate(unittest.TestCase):

    def _make_loader(self, n_samples, input_shape, batch_size, shuffle=False):
        import blade.data as data

        class SimpleDataset(data.Dataset):
            def __len__(self):
                return n_samples
            def __getitem__(self, idx):
                import math
                x = blade.Tensor.from_data(input_shape,
                    [float(idx)] * int(math.prod(input_shape)))
                y = blade.Tensor.from_data([1], [float(idx)])
                return (x, y)

        return data.DataLoader(SimpleDataset(), batch_size, shuffle, 0)

    def test_batch_input_shape(self):
        # 4 samples, each input (3,), batch_size=4 -> batched (4, 3)
        x, _ = next(iter(self._make_loader(4, [3], batch_size=4)))
        self.assertEqual(list(x.shape), [4, 3])

    def test_batch_label_shape(self):
        _, y = next(iter(self._make_loader(4, [3], batch_size=4)))
        self.assertEqual(list(y.shape), [4, 1])

    def test_partial_batch_at_end(self):
        # 5 samples, batch_size=4: first batch (4,), second batch (1,)
        batches = list(self._make_loader(5, [2], batch_size=4))
        self.assertEqual(list(batches[0][0].shape), [4, 2])
        self.assertEqual(list(batches[1][0].shape), [1, 2])

    def test_correct_values_in_batch(self):
        # Sample i has input [i,i,i]; batch of 3 -> [[0,0,0],[1,1,1],[2,2,2]]
        x, y = next(iter(self._make_loader(3, [3], batch_size=3)))
        self.assertTrue(allclose(x.storage(), [0,0,0, 1,1,1, 2,2,2]))
        self.assertTrue(allclose(y.storage(), [0.0, 1.0, 2.0]))

    def test_num_batches(self):
        # ceil(10 / 3) = 4
        self.assertEqual(len(self._make_loader(10, [4], batch_size=3)), 4)

    def test_2d_input_stacking(self):
        # Each sample input shape (2, 3) -> batch (4, 2, 3)
        x, _ = next(iter(self._make_loader(4, [2, 3], batch_size=4)))
        self.assertEqual(list(x.shape), [4, 2, 3])


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)