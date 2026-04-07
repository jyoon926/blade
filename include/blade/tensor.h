#pragma once
#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <stdexcept>

namespace blade {

class Node;

class Tensor {
public:
    Tensor();
    explicit Tensor(std::vector<size_t> shape, bool requires_grad = false);
    Tensor(std::vector<size_t> shape, std::vector<float> data, bool requires_grad = false);

    // Static factories
    static Tensor zeros(std::vector<size_t> shape);
    static Tensor ones(std::vector<size_t> shape);
    static Tensor full(std::vector<size_t> shape, float value);
    static Tensor randn(std::vector<size_t> shape, float mean = 0.f, float std = 1.f);
    static Tensor uniform(std::vector<size_t> shape, float low = 0.f, float high = 1.f);
    static Tensor arange(float start, float stop, float step = 1.f);
    static Tensor from_data(std::vector<size_t> shape, std::vector<float> data);

    // Internal: create a view sharing existing storage (no data copy).
    // Used by reshape/flatten to avoid unnecessary allocations.
    static Tensor _make_view(std::shared_ptr<std::vector<float>> storage,
                             std::vector<size_t> shape,
                             bool requires_grad = false);

    // Shape accessors
    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<size_t>& strides() const { return strides_; }
    size_t ndim() const { return shape_.size(); }
    size_t numel() const;
    size_t size(int dim) const;

    // Data accessors
    float* data_ptr() { return data_->data(); }
    const float* data_ptr() const { return data_->data(); }
    std::vector<float>& storage() { return *data_; }
    const std::vector<float>& storage() const { return *data_; }
    // Returns the underlying shared storage (used by _make_view and share()).
    std::shared_ptr<std::vector<float>> shared_data() const { return data_; }
    float item() const;
    float at(std::vector<size_t> idx) const;
    void set(std::vector<size_t> idx, float value);

    // Autograd
    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool req);
    bool has_grad() const { return grad_ != nullptr; }
    Tensor& grad();
    const Tensor& grad() const;
    void zero_grad();
    void accumulate_grad(const Tensor& g);

    bool is_leaf() const { return is_leaf_; }
    std::shared_ptr<Node> grad_fn() const { return grad_fn_; }
    void set_grad_fn(std::shared_ptr<Node> fn);
    void set_leaf(bool leaf) { is_leaf_ = leaf; }

    // Backward pass entry point
    void backward(const Tensor& grad_output = Tensor());

    // Shape operations (differentiable)
    Tensor reshape(std::vector<size_t> new_shape) const;
    Tensor flatten(int start_dim = 0, int end_dim = -1) const;
    Tensor transpose(int dim0 = 0, int dim1 = 1) const;
    Tensor squeeze(int dim = -1) const;
    Tensor unsqueeze(int dim) const;
    Tensor contiguous() const;
    bool is_contiguous() const;

    // Operator overloads (delegate to ops::)
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator-() const;
    Tensor operator+(float s) const;
    Tensor operator-(float s) const;
    Tensor operator*(float s) const;
    Tensor operator/(float s) const;

    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);

    // Debug
    std::string repr() const;
    void print() const;

private:
    // Shared storage: copies of a Tensor (e.g. via share()) share the same
    // underlying buffer, so saving tensors for backward no longer copies data.
    std::shared_ptr<std::vector<float>> data_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;   // row-major strides in elements
    bool requires_grad_ = false;
    bool is_leaf_ = true;
    std::shared_ptr<Tensor> grad_;
    std::shared_ptr<Node> grad_fn_;

    void compute_strides();
    size_t flat_index(const std::vector<size_t>& idx) const;
};

// Scalar-on-left overloads
Tensor operator+(float s, const Tensor& t);
Tensor operator*(float s, const Tensor& t);

} // namespace blade
