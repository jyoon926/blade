#pragma once
#include <vector>
#include <functional>
#include <memory>
#include <string>

namespace blade {

class Tensor;

// A node in the autograd computation graph.
// Each differentiable op creates a Node that stores the backward function
// and weak references to its input tensors for gradient accumulation.
class Node {
public:
    using BackwardFn = std::function<std::vector<Tensor>(const Tensor& grad_out)>;

    Node(std::string name, BackwardFn fn, std::vector<std::shared_ptr<Tensor>> inputs);

    // Call the backward function and route gradients to inputs.
    void apply(const Tensor& grad_output);

    std::vector<Tensor> compute(const Tensor& grad_output);

    const std::string& name() const { return name_; }
    const std::vector<std::shared_ptr<Tensor>>& inputs() const { return inputs_; }

    // Tensors saved for use in the backward pass (e.g. activations).
    std::vector<Tensor> saved;

private:
    std::string name_;
    BackwardFn backward_fn_;
    std::vector<std::shared_ptr<Tensor>> inputs_;
};

// Execute reverse-mode AD from `root` tensor.
// Performs topological sort and propagates gradients through the graph.
void run_backward(Tensor& root, const Tensor& grad_output);

} // namespace blade
