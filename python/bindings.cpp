#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "blade/tensor.h"
#include "blade/ops.h"
#include "blade/nn/module.h"
#include "blade/nn/layers.h"
#include "blade/nn/activations.h"
#include "blade/nn/loss.h"
#include "blade/nn/pooling.h"
#include "blade/optim/optimizer.h"
#include "blade/optim/sgd.h"
#include "blade/optim/adam.h"
#include "blade/data/dataset.h"
#include "blade/data/mnist.h"

namespace py = pybind11;
using namespace blade;

// Trampoline so Python can subclass nn::Module.
class PyModule : public nn::Module {
public:
    using nn::Module::Module;
    Tensor forward(const Tensor& input) override {
        PYBIND11_OVERRIDE_PURE(Tensor, nn::Module, forward, input);
    }
};

PYBIND11_MODULE(blade, m) {
    m.doc() = "BLADE: Backpropagation Library for Automatic Differentiation and Evaluation";

    // ---- Tensor -------------------------------------------------------------
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<std::vector<size_t>, bool>(),
             py::arg("shape"), py::arg("requires_grad") = false)
        .def(py::init<std::vector<size_t>, std::vector<float>, bool>(),
             py::arg("shape"), py::arg("data"), py::arg("requires_grad") = false)

        // Factories
        .def_static("zeros",   &Tensor::zeros)
        .def_static("ones",    &Tensor::ones)
        .def_static("full",    &Tensor::full)
        .def_static("randn",   &Tensor::randn,
                    py::arg("shape"), py::arg("mean") = 0.f, py::arg("std") = 1.f)
        .def_static("uniform", &Tensor::uniform,
                    py::arg("shape"), py::arg("low") = 0.f, py::arg("high") = 1.f)
        .def_static("arange",  &Tensor::arange,
                    py::arg("start"), py::arg("stop"), py::arg("step") = 1.f)
        .def_static("from_data", &Tensor::from_data)

        // Properties
        .def_property_readonly("shape",  &Tensor::shape)
        .def_property_readonly("ndim",   &Tensor::ndim)
        .def_property_readonly("numel",  &Tensor::numel)
        .def_property("requires_grad",
                      &Tensor::requires_grad,
                      &Tensor::set_requires_grad)
        .def_property_readonly("is_leaf", &Tensor::is_leaf)
        .def_property_readonly("grad",
            [](Tensor& t) -> py::object {
                if (!t.has_grad()) return py::none();
                return py::cast(t.grad());
            })

        // Data access
        .def("item",     &Tensor::item)
        .def("storage",  [](const Tensor& t) { return t.storage(); })
        .def("at",       &Tensor::at)
        .def("set",      &Tensor::set)

        // Autograd
        .def("backward", [](Tensor& t, py::object grad) {
            if (grad.is_none()) t.backward();
            else                t.backward(grad.cast<Tensor>());
        }, py::arg("grad") = py::none())
        .def("zero_grad", &Tensor::zero_grad)

        // Shape ops
        .def("reshape",   &Tensor::reshape)
        .def("flatten",   &Tensor::flatten,
             py::arg("start_dim") = 0, py::arg("end_dim") = -1)
        .def("transpose", &Tensor::transpose,
             py::arg("dim0") = 0, py::arg("dim1") = 1)
        .def("squeeze",   &Tensor::squeeze,  py::arg("dim") = -1)
        .def("unsqueeze", &Tensor::unsqueeze)

        // Operators
        .def("__add__",  [](const Tensor& a, const Tensor& b) { return a + b; })
        .def("__sub__",  [](const Tensor& a, const Tensor& b) { return a - b; })
        .def("__mul__",  [](const Tensor& a, const Tensor& b) { return a * b; })
        .def("__truediv__", [](const Tensor& a, const Tensor& b) { return a / b; })
        .def("__neg__",  [](const Tensor& a) { return -a; })
        .def("__add__",  [](const Tensor& a, float s) { return a + s; })
        .def("__radd__", [](const Tensor& a, float s) { return a + s; })
        .def("__mul__",  [](const Tensor& a, float s) { return a * s; })
        .def("__rmul__", [](const Tensor& a, float s) { return a * s; })

        .def("__repr__", &Tensor::repr)
        .def("print",    &Tensor::print);

    // ---- ops ----------------------------------------------------------------
    py::module_ ops = m.def_submodule("ops");
    ops.def("add",        py::overload_cast<const Tensor&, const Tensor&>(&ops::add));
    ops.def("sub",        py::overload_cast<const Tensor&, const Tensor&>(&ops::sub));
    ops.def("mul",        py::overload_cast<const Tensor&, const Tensor&>(&ops::mul));
    ops.def("div",        py::overload_cast<const Tensor&, const Tensor&>(&ops::div));
    ops.def("neg",        &ops::neg);
    ops.def("pow",        &ops::pow);
    ops.def("sqrt",       &ops::sqrt);
    ops.def("exp",        &ops::exp);
    ops.def("log",        &ops::log);
    ops.def("relu",       &ops::relu);
    ops.def("sigmoid",    &ops::sigmoid);
    ops.def("tanh",       &ops::tanh);
    ops.def("softmax",    &ops::softmax);
    ops.def("log_softmax",&ops::log_softmax);
    ops.def("matmul",     &ops::matmul);
    ops.def("sum",        py::overload_cast<const Tensor&>(&ops::sum));
    ops.def("sum",        py::overload_cast<const Tensor&, int, bool>(&ops::sum),
            py::arg("a"), py::arg("dim"), py::arg("keepdim") = false);
    ops.def("mean",       py::overload_cast<const Tensor&>(&ops::mean));
    ops.def("reshape",    &ops::reshape);
    ops.def("transpose",  &ops::transpose);
    ops.def("flatten",    &ops::flatten,
            py::arg("a"), py::arg("start_dim") = 0, py::arg("end_dim") = -1);
    ops.def("conv2d",     &ops::conv2d,
            py::arg("input"), py::arg("weight"), py::arg("bias"),
            py::arg("stride") = 1, py::arg("padding") = 0);
    ops.def("max_pool2d", &ops::max_pool2d,
            py::arg("input"), py::arg("kernel_size"),
            py::arg("stride") = -1, py::arg("padding") = 0);
    ops.def("dropout",    &ops::dropout);
    ops.def("linear",     &ops::linear);

    // ---- nn -----------------------------------------------------------------
    py::module_ nn = m.def_submodule("nn");

    py::class_<nn::Module, PyModule, std::shared_ptr<nn::Module>>(nn, "Module")
        .def(py::init<>())
        .def("forward",          &nn::Module::forward)
        .def("__call__",         &nn::Module::operator())
        .def("parameters",       &nn::Module::parameters,
             py::return_value_policy::reference_internal)
        .def("train",            &nn::Module::train, py::arg("mode") = true)
        .def("eval",             &nn::Module::eval)
        .def("zero_grad",        &nn::Module::zero_grad)
        .def("save",             &nn::Module::save)
        .def("load",             &nn::Module::load);

    py::class_<nn::Linear, nn::Module, std::shared_ptr<nn::Linear>>(nn, "Linear")
        .def(py::init<size_t, size_t, bool>(),
             py::arg("in_features"), py::arg("out_features"), py::arg("bias") = true)
        .def("forward", &nn::Linear::forward)
        .def_readwrite("weight", &nn::Linear::weight)
        .def_readwrite("bias",   &nn::Linear::bias);

    py::class_<nn::Conv2d, nn::Module, std::shared_ptr<nn::Conv2d>>(nn, "Conv2d")
        .def(py::init<size_t, size_t, size_t, size_t, size_t, bool>(),
             py::arg("in_channels"), py::arg("out_channels"), py::arg("kernel_size"),
             py::arg("stride") = 1, py::arg("padding") = 0, py::arg("bias") = true)
        .def("forward", &nn::Conv2d::forward)
        .def_readwrite("weight", &nn::Conv2d::weight)
        .def_readwrite("bias",   &nn::Conv2d::bias);

    py::class_<nn::BatchNorm2d, nn::Module, std::shared_ptr<nn::BatchNorm2d>>(nn, "BatchNorm2d")
        .def(py::init<size_t, float, float>(),
             py::arg("num_features"), py::arg("eps") = 1e-5f, py::arg("momentum") = 0.1f)
        .def("forward", &nn::BatchNorm2d::forward);

    py::class_<nn::Dropout, nn::Module, std::shared_ptr<nn::Dropout>>(nn, "Dropout")
        .def(py::init<float>(), py::arg("p") = 0.5f)
        .def("forward", &nn::Dropout::forward);

    py::class_<nn::Flatten, nn::Module, std::shared_ptr<nn::Flatten>>(nn, "Flatten")
        .def(py::init<int, int>(),
             py::arg("start_dim") = 1, py::arg("end_dim") = -1)
        .def("forward", &nn::Flatten::forward);

    py::class_<nn::ReLU, nn::Module, std::shared_ptr<nn::ReLU>>(nn, "ReLU")
        .def(py::init<>()).def("forward", &nn::ReLU::forward);

    py::class_<nn::Sigmoid, nn::Module, std::shared_ptr<nn::Sigmoid>>(nn, "Sigmoid")
        .def(py::init<>()).def("forward", &nn::Sigmoid::forward);

    py::class_<nn::Tanh, nn::Module, std::shared_ptr<nn::Tanh>>(nn, "Tanh")
        .def(py::init<>()).def("forward", &nn::Tanh::forward);

    py::class_<nn::Softmax, nn::Module, std::shared_ptr<nn::Softmax>>(nn, "Softmax")
        .def(py::init<int>(), py::arg("dim") = -1)
        .def("forward", &nn::Softmax::forward);

    py::class_<nn::MaxPool2d, nn::Module, std::shared_ptr<nn::MaxPool2d>>(nn, "MaxPool2d")
        .def(py::init<size_t, size_t, size_t>(),
             py::arg("kernel_size"), py::arg("stride") = 0, py::arg("padding") = 0)
        .def("forward", &nn::MaxPool2d::forward);

    // Loss functions (not Modules – they take two tensors)
    nn.def("cross_entropy", &nn::cross_entropy);
    nn.def("mse_loss",      &nn::mse_loss);
    nn.def("nll_loss",      &nn::nll_loss);

    py::class_<nn::CrossEntropyLoss>(nn, "CrossEntropyLoss")
        .def(py::init<>())
        .def("__call__", &nn::CrossEntropyLoss::operator());

    py::class_<nn::MSELoss>(nn, "MSELoss")
        .def(py::init<>())
        .def("__call__", &nn::MSELoss::operator());

    // ---- optim --------------------------------------------------------------
    py::module_ optim = m.def_submodule("optim");

    py::class_<optim::Optimizer>(optim, "Optimizer")
        .def("zero_grad", &optim::Optimizer::zero_grad);

    py::class_<optim::SGD, optim::Optimizer>(optim, "SGD")
        .def(py::init<std::vector<Tensor*>, float, float, float, bool>(),
             py::arg("params"), py::arg("lr"),
             py::arg("momentum") = 0.f, py::arg("weight_decay") = 0.f,
             py::arg("nesterov") = false)
        .def("step",      &optim::SGD::step)
        .def("zero_grad", &optim::SGD::zero_grad);

    py::class_<optim::Adam, optim::Optimizer>(optim, "Adam")
        .def(py::init<std::vector<Tensor*>, float, float, float, float, float>(),
             py::arg("params"), py::arg("lr") = 1e-3f,
             py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f,
             py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.f)
        .def("step",      &optim::Adam::step)
        .def("zero_grad", &optim::Adam::zero_grad);

    // ---- data ---------------------------------------------------------------
    py::module_ data = m.def_submodule("data");

    py::class_<data::Dataset, std::shared_ptr<data::Dataset>>(data, "Dataset")
        .def("__len__", &data::Dataset::size)
        .def("__getitem__", &data::Dataset::get);

    py::class_<data::DataLoader>(data, "DataLoader")
        .def(py::init<std::shared_ptr<data::Dataset>, size_t, bool, size_t>(),
             py::arg("dataset"), py::arg("batch_size"),
             py::arg("shuffle") = true, py::arg("num_workers") = 0)
        .def("__len__",  &data::DataLoader::num_batches)
        .def("__iter__", [](data::DataLoader& dl) {
            return py::make_iterator(dl.begin(), dl.end());
        }, py::keep_alive<0, 1>());

    py::enum_<data::MNIST::Split>(data, "MNISTSplit")
        .value("Train", data::MNIST::Split::Train)
        .value("Test",  data::MNIST::Split::Test)
        .export_values();

    py::class_<data::MNIST, data::Dataset, std::shared_ptr<data::MNIST>>(data, "MNIST")
        .def(py::init<const std::string&, data::MNIST::Split, bool>(),
             py::arg("root"),
             py::arg("split") = data::MNIST::Split::Train,
             py::arg("download") = false)
        .def("__len__",     &data::MNIST::size)
        .def("__getitem__", &data::MNIST::get);
}
