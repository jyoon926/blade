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

void Node::apply(const Tensor& grad_output) {
    if (!backward_fn_)
        throw std::runtime_error("Node '" + name_ + "' has no backward function");

    std::vector<Tensor> grad_inputs = backward_fn_(grad_output);

    if (grad_inputs.size() != inputs_.size())
        throw std::runtime_error("Node '" + name_ + "': grad count mismatch");

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
    // Initialise root gradient: ones if not provided (default-constructed Tensor has empty storage).
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
        // No computation graph – nothing to do.
        return;
    }

    // Map each node to its accumulated upstream gradient.
    std::unordered_map<Node*, Tensor> node_grads;
    node_grads[root.grad_fn().get()] = g;

    auto order = topo_sort(root);
    for (Node* node : order) {
        auto it = node_grads.find(node);
        if (it == node_grads.end()) continue;

        const Tensor& upstream = it->second;
        node->apply(upstream);

        // Leaf gradients are accumulated inside Node::apply above.
    }
}

} // namespace blade
