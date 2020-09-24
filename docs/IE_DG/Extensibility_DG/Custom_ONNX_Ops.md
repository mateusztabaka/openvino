# Custom ONNX operators {#openvino_docs_IE_DG_Extensibility_DG_Custom_ONNX_Ops}

ONNX importer provides mechanism to register custom ONNX operators based on predefined or user defined nGraph operations.
The function responsible for registering a new operator is called `ngraph::onnx_import::register_operator` and is defined in `onnx_import/onnx_utils.hpp`.

## Example 1: registering custom ONNX operator based on predefined nGraph operations

As an example, let's register 'CustomRelu' ONNX operator in a domain called 'custom_domain'.
CustomRelu is defined as follows:
```
x >= 0 => f(x) = x * alpha
x < 0  => f(x) = x * beta
```
where alpha, beta are float constants.

Let's start with including headers:
@snippet docs/onnx_custom_op/main.cpp onnx_custom_op:headers

Next we define a node with op_type 'CustomRelu', domain 'custom_domain', one input, one output and two float attributes 'alpha' and 'beta'.
@snippet docs/onnx_custom_op/main.cpp onnx_custom_op:model

Before the model can be read by Inference Engine, we need to register 'CustomRelu' operator in ONNX importer:
@snippet docs/onnx_custom_op/main.cpp onnx_custom_op:register_operator
`register_operator` function takes four arguments: op_type, opset version, domain, and a function object.
The function object is a user defined function that takes `ngraph::onnx_import::Node` as an input and based on that, returns a graph with nGraph operations.
Class `ngraph::onnx_import::Node` represents a node in ONNX model. It provides, functions to fetch input node(s) (`get_ng_inputs`), fetch attribute value (`get_attribute_value`) and many more (please refer to `onnx_import/core/node.hpp` for full class declaration).

For a reference on how to create a graph with nGraph operations, visit [nGraph tutorial](../../nGraphTutorial.md).
For a complete list of predefined nGraph operators, visit [available operations sets](../../ops/opset.md).

New operator registration must happen before the ONNX model is read, e.g. if a ONNX model uses operator 'CustomRelu', `register_operator("CustomRelu", ...)` must be called before InferenceEngine::Core::ReadNetwork.

Re-registering ONNX operators within the same process is supported. During registration of existing operator, a warning is printed.

## Requirements for building with CMake

Program that uses `register_operator` functionality, requires (in addition to Inference Engine) `ngraph` and `onnx_importer` libraries.
Both `ngraph` and `onnx_importer` libraries are under single package namespace, so `find_package(ngraph REQUIRED)` is sufficient to find them both.
`ngraph` package exposes two variables (`${NGRAPH_LIBRARIES}` and `${ONNX_IMPORTER_LIBRARIES}`) which reference `ngraph` and `onnx_importer` libraries.
Those variables need to be passed to `target_link_libraries` command in CMakeLists.txt file.

See below `CMakeLists.txt` for reference:
@snippet docs/onnx_custom_op/CMakeLists.txt cmake:onnx_custom_op

## Example 2: registering custom ONNX operator based on custom nGraph operations
