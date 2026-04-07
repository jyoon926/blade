#include "blade/nn/module.h"
#include <fstream>
#include <stdexcept>
#include <unordered_map>

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

// ---- Serialisation ----------------------------------------------------------
// Binary format (little-endian native floats):
//   [n_params: size_t]
//   for each param:
//     [name_len: size_t] [name: char * name_len]
//     [ndim: size_t]     [shape: size_t * ndim]
//     [data: float * numel]

void Module::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Module::save: cannot open '" + path + "'");

    auto params = const_cast<Module*>(this)->named_parameters();
    size_t n = params.size();
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));

    for (auto& [name, p] : params) {
        size_t name_len = name.size();
        f.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        f.write(name.data(), (std::streamsize)name_len);

        const auto& shape = p->shape();
        size_t ndim = shape.size();
        f.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
        f.write(reinterpret_cast<const char*>(shape.data()), (std::streamsize)(ndim * sizeof(size_t)));

        size_t numel = p->numel();
        f.write(reinterpret_cast<const char*>(p->data_ptr()), (std::streamsize)(numel * sizeof(float)));
    }
}

void Module::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Module::load: cannot open '" + path + "'");

    // Build name → Tensor* map for O(1) lookup.
    std::unordered_map<std::string, Tensor*> param_map;
    for (auto& [name, p] : named_parameters())
        param_map[name] = p;

    size_t n;
    f.read(reinterpret_cast<char*>(&n), sizeof(n));

    for (size_t i = 0; i < n; ++i) {
        size_t name_len;
        f.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        std::string name(name_len, '\0');
        f.read(name.data(), (std::streamsize)name_len);

        size_t ndim;
        f.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
        std::vector<size_t> shape(ndim);
        f.read(reinterpret_cast<char*>(shape.data()), (std::streamsize)(ndim * sizeof(size_t)));

        size_t numel = 1;
        for (auto d : shape) numel *= d;

        auto it = param_map.find(name);
        if (it == param_map.end()) {
            // Unknown parameter — skip its data.
            f.seekg((std::streamoff)(numel * sizeof(float)), std::ios::cur);
            continue;
        }
        if (it->second->numel() != numel)
            throw std::runtime_error("Module::load: shape mismatch for '" + name + "'");
        f.read(reinterpret_cast<char*>(it->second->data_ptr()),
               (std::streamsize)(numel * sizeof(float)));
    }
}

} // namespace nn
} // namespace blade
