#include "blade/tensor.h"
#include "blade/node.h"
#include "blade/ops.h"
#include <omp.h>
#include <cmath>
#include <cstdio>
#include <cassert>
#include <random>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <sstream>

#define NUM_THREADS 8

namespace blade {

// ---- helpers ----------------------------------------------------------------

static size_t shape_numel(const std::vector<size_t>& s) {
    size_t n = 1;
    for (auto d : s) n *= d;
    return n;
}

void Tensor::compute_strides() {
    strides_.resize(shape_.size());
    size_t stride = 1;
    for (int i = (int)shape_.size() - 1; i >= 0; --i) {
        strides_[i] = stride;
        stride *= shape_[i];
    }
}

size_t Tensor::flat_index(const std::vector<size_t>& idx) const {
    if (idx.size() != shape_.size())
        throw std::runtime_error("flat_index: rank mismatch");
    size_t fi = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (idx[i] >= shape_[i])
            throw std::out_of_range("flat_index: index out of range");
        fi += idx[i] * strides_[i];
    }
    return fi;
}

// ---- constructors -----------------------------------------------------------

Tensor::Tensor()
    : data_(std::make_shared<std::vector<float>>()) {}

Tensor::Tensor(std::vector<size_t> shape, bool requires_grad)
    : data_(std::make_shared<std::vector<float>>()),
      shape_(std::move(shape)), requires_grad_(requires_grad) {
    compute_strides();
    data_->assign(numel(), 0.f);
}

Tensor::Tensor(std::vector<size_t> shape, std::vector<float> data, bool requires_grad)
    : data_(std::make_shared<std::vector<float>>(std::move(data))),
      shape_(std::move(shape)), requires_grad_(requires_grad) {
    if (data_->size() != shape_numel(shape_))
        throw std::runtime_error("Tensor: data/shape size mismatch");
    compute_strides();
}

// ---- static factories -------------------------------------------------------

Tensor Tensor::zeros(std::vector<size_t> shape) {
    return Tensor(shape);
}

Tensor Tensor::ones(std::vector<size_t> shape) {
    Tensor t(shape);
    std::fill(t.data_->begin(), t.data_->end(), 1.f);
    return t;
}

Tensor Tensor::full(std::vector<size_t> shape, float value) {
    Tensor t(shape);
    std::fill(t.data_->begin(), t.data_->end(), value);
    return t;
}

Tensor Tensor::randn(std::vector<size_t> shape, float mean, float std) {
    Tensor t(shape);
    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<float> dist(mean, std);
    for (auto& v : *t.data_) v = dist(rng);
    return t;
}

Tensor Tensor::uniform(std::vector<size_t> shape, float low, float high) {
    Tensor t(shape);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(low, high);
    for (auto& v : *t.data_) v = dist(rng);
    return t;
}

Tensor Tensor::arange(float start, float stop, float step) {
    size_t n = (size_t)std::ceil((stop - start) / step);
    Tensor t({n});
    for (size_t i = 0; i < n; ++i) t.data_ptr()[i] = start + i * step;
    return t;
}

Tensor Tensor::from_data(std::vector<size_t> shape, std::vector<float> data) {
    return Tensor(std::move(shape), std::move(data));
}

Tensor Tensor::_make_view(std::shared_ptr<std::vector<float>> storage,
                          std::vector<size_t> shape,
                          bool requires_grad) {
    Tensor t;
    t.data_         = std::move(storage);
    t.shape_        = std::move(shape);
    t.requires_grad_ = requires_grad;
    t.compute_strides();
    return t;
}

// ---- accessors --------------------------------------------------------------

size_t Tensor::numel() const { return shape_numel(shape_); }

size_t Tensor::size(int dim) const {
    if (dim < 0) dim += (int)shape_.size();
    return shape_.at((size_t)dim);
}

float Tensor::item() const {
    if (numel() != 1)
        throw std::runtime_error("item() called on non-scalar tensor");
    return (*data_)[0];
}

float Tensor::at(std::vector<size_t> idx) const {
    return (*data_)[flat_index(idx)];
}

void Tensor::set(std::vector<size_t> idx, float value) {
    (*data_)[flat_index(idx)] = value;
}

// ---- autograd ---------------------------------------------------------------

void Tensor::set_requires_grad(bool req) {
    requires_grad_ = req;
    if (req && !grad_) {
        grad_ = std::make_shared<Tensor>(shape_);
    }
}

Tensor& Tensor::grad() {
    if (!grad_) throw std::runtime_error("Tensor has no gradient");
    return *grad_;
}

const Tensor& Tensor::grad() const {
    if (!grad_) throw std::runtime_error("Tensor has no gradient");
    return *grad_;
}

void Tensor::zero_grad() {
    if (grad_) std::fill(grad_->data_->begin(), grad_->data_->end(), 0.f);
}

void Tensor::accumulate_grad(const Tensor& g) {
    if (!grad_) grad_ = std::make_shared<Tensor>(shape_);
    const size_t n = numel();
    float*       dst = grad_->data_ptr();
    const float* src = g.data_ptr();
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (size_t i = 0; i < n; ++i)
        dst[i] += src[i];
}

void Tensor::set_grad_fn(std::shared_ptr<Node> fn) {
    grad_fn_ = std::move(fn);
    is_leaf_ = false;
}

void Tensor::backward(const Tensor& grad_output) {
    run_backward(*this, grad_output);
}

// ---- shape operations -------------------------------------------------------

bool Tensor::is_contiguous() const { return true; }

Tensor Tensor::contiguous() const { return *this; }

Tensor Tensor::reshape(std::vector<size_t> new_shape) const {
    return ops::reshape(*this, std::move(new_shape));
}

Tensor Tensor::flatten(int start_dim, int end_dim) const {
    return ops::flatten(*this, start_dim, end_dim);
}

Tensor Tensor::transpose(int dim0, int dim1) const {
    return ops::transpose(*this, dim0, dim1);
}

Tensor Tensor::squeeze(int dim) const {
    return ops::squeeze(*this, dim);
}

Tensor Tensor::unsqueeze(int dim) const {
    return ops::unsqueeze(*this, dim);
}

// ---- operators --------------------------------------------------------------

Tensor Tensor::operator+(const Tensor& o) const { return ops::add(*this, o); }
Tensor Tensor::operator-(const Tensor& o) const { return ops::sub(*this, o); }
Tensor Tensor::operator*(const Tensor& o) const { return ops::mul(*this, o); }
Tensor Tensor::operator/(const Tensor& o) const { return ops::div(*this, o); }
Tensor Tensor::operator-() const              { return ops::neg(*this); }
Tensor Tensor::operator+(float s) const       { return ops::add(*this, s); }
Tensor Tensor::operator-(float s) const       { return ops::sub(*this, s); }
Tensor Tensor::operator*(float s) const       { return ops::mul(*this, s); }
Tensor Tensor::operator/(float s) const       { return ops::div(*this, s); }

Tensor& Tensor::operator+=(const Tensor& other) {
    // In-place; breaks the graph — only safe for leaf parameter updates.
    const size_t n  = numel();
    float*       d  = data_ptr();
    const float* od = other.data_ptr();
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (size_t i = 0; i < n; ++i) d[i] += od[i];
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    const size_t n  = numel();
    float*       d  = data_ptr();
    const float* od = other.data_ptr();
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
    for (size_t i = 0; i < n; ++i) d[i] -= od[i];
    return *this;
}

Tensor operator+(float s, const Tensor& t) { return ops::add(t, s); }
Tensor operator*(float s, const Tensor& t) { return ops::mul(t, s); }

// ---- debug ------------------------------------------------------------------

std::string Tensor::repr() const {
    std::ostringstream ss;
    ss << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        ss << shape_[i];
        if (i + 1 < shape_.size()) ss << ", ";
    }
    ss << "], requires_grad=" << (requires_grad_ ? "true" : "false") << ")";
    return ss.str();
}

void Tensor::print() const {
    printf("%s\n", repr().c_str());
}

} // namespace blade
