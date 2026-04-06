#include "blade/node.h"
#include "blade/tensor.h"
#include <unordered_map>
#include <vector>
#include <queue>
#include <stdexcept>

namespace blade {

Node::Node(std::string name, BackwardFn fn,
           std::vector<std::shared_ptr<Tensor>> inputs)
    : name_(std::move(name)),
      backward_fn_(std::move(fn)),
      inputs_(std::move(inputs)) {}

std::vector<Tensor> Node::compute(const Tensor& grad_output) {
    if (!backward_fn_)
        throw std::runtime_error("Node '" + name_ + "' has no backward function");
    std::vector<Tensor> grad_inputs = backward_fn_(grad_output);
    if (grad_inputs.size() != inputs_.size())
        throw std::runtime_error("Node '" + name_ + "': grad count mismatch");
    return grad_inputs;
}

void Node::apply(const Tensor& grad_output) {
    std::vector<Tensor> grad_inputs = compute(grad_output);
    for (size_t i = 0; i < inputs_.size(); ++i) {
        auto& inp = inputs_[i];
        if (!inp || !inp->requires_grad()) continue;
        inp->accumulate_grad(grad_inputs[i]);
    }
}

// ---- reverse-mode AD --------------------------------------------------------

// Build reverse topological order of nodes reachable from `root`.
static std::vector<Node*> topo_sort(Tensor& root) {
    std::vector<Node*> order;
    std::unordered_map<Node*, bool> visited;

    std::function<void(Node*)> dfs = [&](Node* node) {
        if (!node || visited[node]) return;
        visited[node] = true;
        for (auto& inp : node->inputs()) {
            if (inp && inp->grad_fn())
                dfs(inp->grad_fn().get());
        }
        order.push_back(node);
    };

    if (root.grad_fn()) dfs(root.grad_fn().get());
    // order is in post-order (leaves first) – reverse for correct backprop order
    std::reverse(order.begin(), order.end());
    return order;
}

void run_backward(Tensor& root, const Tensor& grad_output) {
    Tensor g;
    if (grad_output.storage().empty()) {
        g = Tensor::ones(root.shape());
    } else {
        g = grad_output;
    }

    if (root.requires_grad() && root.is_leaf()) {
        root.accumulate_grad(g);
        return;
    }

    if (!root.grad_fn()) {
        return;
    }

    // Map each node to its accumulated upstream gradient.
    std::unordered_map<Node*, Tensor> node_grads;
    node_grads[root.grad_fn().get()] = g;

    auto order = topo_sort(root);
    for (Node* node : order) {
        auto it = node_grads.find(node);
        if (it == node_grads.end()) continue;

        // Compute raw gradients for each input of this node.
        std::vector<Tensor> grad_inputs = node->compute(it->second);

        for (size_t i = 0; i < node->inputs().size(); ++i) {
            auto& inp = node->inputs()[i];
            if (!inp || !inp->requires_grad()) continue;

            if (inp->is_leaf()) {
                // Leaf tensor: accumulate gradient directly.
                inp->accumulate_grad(grad_inputs[i]);
            } else if (inp->grad_fn()) {
                // Non-leaf: pass gradient to the next node in the chain.
                // Use element-wise addition to handle multi-path graphs.
                Node* inp_fn = inp->grad_fn().get();
                auto ins_it = node_grads.find(inp_fn);
                if (ins_it == node_grads.end()) {
                    node_grads[inp_fn] = grad_inputs[i];
                } else {
                    const size_t n = ins_it->second.numel();
                    for (size_t j = 0; j < n; ++j)
                        ins_it->second.data_ptr()[j] += grad_inputs[i].data_ptr()[j];
                }
            }
        }
    }
}

} // namespace blade
