#include "blade/nn/module.h"
#include <fstream>
#include <stdexcept>

namespace blade {
namespace nn {

Tensor Module::operator()(const Tensor& input) {
    return forward(input);
}

void Module::register_parameter(const std::string& name, Tensor& param) {
    param.set_requires_grad(true);
    params_.push_back({name, &param});
}

void Module::register_module(const std::string& name, std::shared_ptr<Module> mod) {
    mod->set_name(name);
    submodules_.push_back({name, std::move(mod)});
}

std::vector<Tensor*> Module::parameters() {
    std::vector<Tensor*> result;
    for (auto& [name, p] : params_)
        result.push_back(p);
    for (auto& [name, mod] : submodules_) {
        auto sub_params = mod->parameters();
        result.insert(result.end(), sub_params.begin(), sub_params.end());
    }
    return result;
}

std::vector<std::pair<std::string, Tensor*>> Module::named_parameters() {
    std::vector<std::pair<std::string, Tensor*>> result;
    std::string prefix = name_.empty() ? "" : name_ + ".";
    for (auto& [n, p] : params_)
        result.push_back({prefix + n, p});
    for (auto& [n, mod] : submodules_) {
        auto sub = mod->named_parameters();
        result.insert(result.end(), sub.begin(), sub.end());
    }
    return result;
}

std::vector<std::shared_ptr<Module>> Module::children() {
    std::vector<std::shared_ptr<Module>> result;
    for (auto& [n, m] : submodules_) result.push_back(m);
    return result;
}

std::vector<std::pair<std::string, std::shared_ptr<Module>>> Module::named_children() {
    return submodules_;
}

void Module::train(bool mode) {
    training_ = mode;
    for (auto& [n, m] : submodules_) m->train(mode);
}

void Module::eval() { train(false); }

void Module::zero_grad() {
    for (auto* p : parameters()) p->zero_grad();
}

void Module::save(const std::string& path) const {
    // TODO: binary serialisation of all named parameters
    (void)path;
}

void Module::load(const std::string& path) {
    // TODO: load and restore named parameters
    (void)path;
}

} // namespace nn
} // namespace blade
