#include "blade/ops.h"
#include "internal.h"

namespace blade {
namespace ops {

// reshape is zero-copy: the output shares the input's storage buffer.
// Only shape metadata differs; a backward view-reshape undoes it.
Tensor reshape(const Tensor& a, std::vector<size_t> new_shape) {
    size_t new_n = 1;
    for (auto d : new_shape) new_n *= d;
    if (new_n != a.numel()) throw std::runtime_error("reshape: size mismatch");
    Tensor out = Tensor::_make_view(a.shared_data(), new_shape);
    if (a.requires_grad()) {
        out.set_requires_grad(true);
        auto sa = share(a);
        out.set_grad_fn(std::make_shared<Node>("ReshapeBackward",
            [sa](const Tensor& g) -> std::vector<Tensor> {
                return {reshape(g, sa->shape())};
            }, std::vector<std::shared_ptr<Tensor>>{sa}));
    }
    return out;
}

Tensor transpose(const Tensor& a, int dim0, int dim1) {
    int ndim = (int)a.ndim();
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;
    if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim)
        throw std::runtime_error("transpose: invalid dims");
    std::vector<size_t> new_shape = a.shape();
    std::swap(new_shape[dim0], new_shape[dim1]);
    Tensor out(new_shape);
    if (ndim == 2) {
        size_t R = a.shape()[0], C = a.shape()[1];
        #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
        for (size_t i = 0; i < R; ++i)
            for (size_t j = 0; j < C; ++j)
                out.data_ptr()[j * R + i] = a.data_ptr()[i * C + j];
    }
    if (a.requires_grad()) {
        out.set_requires_grad(true);
        auto sa = share(a);
        out.set_grad_fn(std::make_shared<Node>("TransposeBackward",
            [sa, dim0, dim1](const Tensor& g) -> std::vector<Tensor> {
                return {transpose(g, dim0, dim1)};
            }, std::vector<std::shared_ptr<Tensor>>{sa}));
    }
    return out;
}

Tensor squeeze(const Tensor& a, int dim) {
    std::vector<size_t> new_shape;
    for (int i = 0; i < (int)a.ndim(); ++i) {
        if (dim < 0 || i == dim) {
            if (a.shape()[i] != 1) new_shape.push_back(a.shape()[i]);
        } else {
            new_shape.push_back(a.shape()[i]);
        }
    }
    return reshape(a, new_shape);
}

Tensor unsqueeze(const Tensor& a, int dim) {
    if (dim < 0) dim += (int)a.ndim() + 1;
    std::vector<size_t> new_shape = a.shape();
    new_shape.insert(new_shape.begin() + dim, 1);
    return reshape(a, new_shape);
}

Tensor flatten(const Tensor& a, int start_dim, int end_dim) {
    int ndim = (int)a.ndim();
    if (start_dim < 0) start_dim += ndim;
    if (end_dim   < 0) end_dim   += ndim;
    size_t flat = 1;
    std::vector<size_t> new_shape;
    for (int i = 0; i < ndim; ++i) {
        if (i >= start_dim && i <= end_dim) {
            flat *= a.shape()[i];
            if (i == end_dim) new_shape.push_back(flat);
        } else {
            new_shape.push_back(a.shape()[i]);
        }
    }
    return reshape(a, new_shape);
}

} // namespace ops
} // namespace blade
