#pragma once
#include "blade/tensor.h"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace blade {
namespace nn {

class Module {
public:
    virtual ~Module() = default;

    // Subclasses implement the computation.
    virtual Tensor forward(const Tensor& input) = 0;
    Tensor operator()(const Tensor& input);

    // Parameter access
    std::vector<Tensor*> parameters();
    std::vector<std::pair<std::string, Tensor*>> named_parameters();

    // Submodule access
    std::vector<std::shared_ptr<Module>> children();
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> named_children();

    // Training / eval mode (propagates to all submodules)
    void train(bool mode = true);
    void eval();
    bool training() const { return training_; }

    // Zero all parameter gradients
    void zero_grad();

    // Serialisation (saves/loads all named parameters)
    void save(const std::string& path) const;
    void load(const std::string& path);

    std::string name() const { return name_; }
    void set_name(const std::string& n) { name_ = n; }

    // Register params and submodules (called from constructors or Python __init__).
    void register_parameter(const std::string& name, Tensor& param);
    void register_module(const std::string& name, std::shared_ptr<Module> mod);

protected:

    bool training_ = true;

private:
    std::string name_;
    std::vector<std::pair<std::string, Tensor*>> params_;
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> submodules_;
};

} // namespace nn
} // namespace blade
