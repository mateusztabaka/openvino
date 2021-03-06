# Writing ngraph transformations {#new_ngraph_transformation}

This guide contains all necessary information that could help you to start writing nGraph transformations.

First of all before writing transformation make sure that there is no transformation with the same functionality
in [Transformation Library](group__ie__transformation__api.html). To start writing transformation it's good to know
how [Transformation Library](group__ie__transformation__api.html) is structured, how transformations are organized
and where to put your transformation code.

Let's start from reviewing transformations library structure.
Transformations library is independent from InferenceEngine target library named as `inference_engine_transformations`
and located in `inference-engine/src/transformations` directory.

Transformations root directory contains two folders:
1. ngraph_ops - legacy opset operations needed for nGraph to CNNNetwork conversion.
> **Note**: this operation are prohibited to use inside new plugins until they are not moved to separate directory with allowed operations.
2. transformations - includes all transformations, utils, runtime info attributes and pass managers.
> **Note**: do not use transformation that belongs to `ngraph::pass::ConvertOpSet1ToLegacy` transformations until they are not moved to separate directory with allowed transformations.

Transformation flow in transformation library has several layers:
1. Pass managers - executes any type of transformations and provides additional debug capabilities.
2. Transformations - performs particular transformation algorithm on `ngraph::Function`.
3. Low level functions that takes set of nodes and performs some transformation action. 
They are not mandatory and all transformation code can be located inside transformation.
But if some transformation parts can potentially be reused in other transformations we suggest to keep them as a separate functions.

To decide where to store your transformation code please follow these rules:
1. If it's plugin specific transformation and can't be reused by other plugins keep source code inside plugin.
2. If this transformation relates to OpSetXToOpSetY conversion or it's common optimization then keep sources inside transformation library.

After you decided where to store your transformation code you can start develop your own nGraph transformation.

## Table of Contents:

### 1. [`ngraph::Function` and graph representation](#ngraph_function) 
### 2. [Transformations types](#transformations_types)
### 2.1 [Function pass](#function_pass)
### 2.2 [Matcher pass](#matcher_pass)
### 2.3 [GraphRewrite pass](#graph_rewrite_pass) 
### 3. [Pattern matching](#pattern_matching)
### 4. [Working with ngraph::Function](#working_with_ngraph_function)
### 5. [Transformation writing essentials](#transformation_writing_essentials)
### 6. [Common mistakes in transformations](#common_mistakes)
### 7. [Using pass manager](#using_pass_manager)
### 8. [How to debug transformations](#how_to_debug_transformations)
### 9. [Disabling/Enabling specific transformations for plugin X](#disabling_transformation)
### 10. [Transformations testing](#transformations_testing)

## ngraph::Function and graph representation <a name="ngraph_function"></a>

nGraph function is a very simple thing: it stores shared pointers to `ngraph::op::Result` and `ngraph::op::Parameter` operations that are inputs and outputs of the graph. 
All other operations hold each other via shared pointers: child operation holds its parent (hard link). If operation has no consumers and it's not Result operation
(shared pointer counter is zero) then it will be destructed and won't be accessible anymore. Each operation in `ngraph::Function` has a `std::shared_ptr<ngraph::Node>` type.

Below you can find examples how `ngraph::Function` can be created:

@snippet example_ngraph_utils.cpp ngraph_utils:simple_function

@snippet example_ngraph_utils.cpp ngraph_utils:advanced_function

## Transformations types <a name="transformations_types"></a>

nGraph has tree main transformation types: `ngraph::pass::FunctionPass` - strait forward way to work with `ngraph::Function` directly;
`ngraph::pass::MatcherPass` - pattern based transformation approach; `ngraph::pass::GraphRewrite` - container for matcher passes.

![transformations_structure]

###1. ngraph::pass::FunctionPass <a name="function_pass"></a>

`ngraph::pass::FunctionPass` is used for transformations that take entire `ngraph::Function` as input and process it.

Template for FunctionPass transformation class

@snippet src/template_function_transformation.hpp function_pass:template_transformation_hpp

@snippet src/template_function_transformation.cpp function_pass:template_transformation_cpp

Using `ngraph::FunctionPass` you need to override `run_on_function` method where you will write transformation code.
Return value must be `true` if original function has changed during transformation (new operation were added or operations replacement was made or node attributes were changed) otherwise it must be `false`.
For transformation API please follow [working with ngraph::Function](#working_with_ngraph_function) section.
Also `ngraph::FunctionPass` based transformations can be executed via `pass::Manager`. See examples in [Using pass manager](#using_pass_manager) section.

###2. ngraph::pass::MatcherPass <a name="matcher_pass"></a>

`ngraph::pass::MatcherPass` is used for pattern based transformations.

Template for MatcherPass transformation class
@snippet src/template_pattern_transformation.hpp graph_rewrite:template_transformation_hpp

@snippet src/template_pattern_transformation.cpp graph_rewrite:template_transformation_cpp

Using `ngraph::pass::MatcherPass` you need to complete these steps:
1. Create pattern
2. Implement callback 
3. Register pattern and Matcher
4. MatcherPass execution

So let's go though each of this steps.

### Create pattern
Pattern is a single root `ngraph::Function`. But the only difference is that you don't need to create function object, you just create and connect nGraph or special pattern operations.
And then take the last created operation and put it as a root of the pattern. This root node will be used as a root node in pattern matching.
> **Note**: any nodes in pattern that have no consumers and not registered as root won't be used in pattern matching. 

@snippet example_ngraph_utils.cpp pattern:simple_example

You may have noticed that `Parameter` operation in example has type and shape specified. These attributes are needed only to create Parameter operation class and won't be used in pattern matching. 

But what if we want to match pattern where `ShapeOf` takes any operation as input? To find an answer please follow [pattern matching](#pattern_matching) section.

### Implement callback
Callback is an action applied to every pattern entrance. In general callback is lambda function that takes Matcher object with detected sub-graph.

@snippet example_ngraph_utils.cpp pattern:callback_example

Example above shows callback structure and how Matcher can be used for accessing nodes detected by pattern.
Callback return value must be `true` if root node was replaced and another pattern can't be applied to the same root node otherwise it must be `false`.
> **Note**: it's not recommended to manipulate with nodes that are under root node. This may affect GraphRewrite execution as it's expected that all nodes that comes after root node in topological order are valid and can be used in pattern matching. 

MatcherPass also provides functionality that allows to report which newly created nodes can be used in additional pattern matching.
If MatcherPass was registered in `pass::Manager` or `pass::GraphRewrite` then this registered nodes will be added for additional pattern matching.
That means that matcher passes registered in `pass::GraphRewrite` will be applied to this nodes.

Example below shows how single MatcherPass can fuse sequence of operations using `register_new_node` method.

@snippet src/template_pattern_transformation.cpp matcher_pass:relu_fusion

> **Note**: if you register multiple nodes please add them in topological order. We do not topologically sort this nodes as it's time consuming operation.

### Register pattern and Matcher
The last step is to register Matcher and callback inside MatcherPass pass. And to do this you need to call `register_matcher` method.
> **Note**: Only one matcher can be registered for single MatcherPass class.

```cpp
// Register matcher and callback
register_matcher(m, callback);
```
### Matcher pass execution
MatcherPass has multiple ways to be executed:
1. Run on a single node - it can be useful if you want to run MatcherPass inside another transformation.
@snippet src/template_pattern_transformation.cpp matcher_pass:run_on_node
2. Run on `ngraph::Function` using GraphRewrite - this approach gives ability to run MatcherPass on whole `ngraph::Functoin`. Moreover multiple MatcherPass transformation can be registered in a single GraphRewite to be executed in a single graph traversal.
@snippet src/template_pattern_transformation.cpp matcher_pass:graph_rewrite
3. Run on `ngraph::Function` using `pass::Manager` - this approach helps you to register MatcherPass for execution on `ngraph::Function` as another transformation types.
@snippet src/template_pattern_transformation.cpp matcher_pass:manager


###3. ngraph::pass::GraphRewrite <a name="graph_rewrite_pass"></a>

GraphRewrite pass serves for running multiple matcher passes on `ngraph::Function` in a single graph traversal. 
Example:

@snippet src/template_pattern_transformation.cpp matcher_pass:graph_rewrite

In addition GraphRewrite handles nodes that were registered by MatcherPasses during their execution. This nodes will be added to the beginning of sequence with nodes for pattern matching.

> **Note**: when using `pass::Manager` temporary GraphRewrite is used to execute single MatcherPass. 

GraphRewrite has two algorithms for MatcherPasses execution. First algorithm is a straight-forward. It applies each MatcherPass in registraion order to current node.

![graph_rewrite_execution]

But it is nor really efficient when you have a lot of registered passes. So first of all GraphRewrite check that all MatcherPass patterns has type based root node (it means that type of this node is not hidden into predicate).
And then creates map from registered MatcherPases. That helps to avoid additional cost of applying each MatcherPass for each node.

![graph_rewrite_efficient_search] 

## Pattern matching <a name="pattern_matching"></a>

Sometimes patterns can't be expressed via regular nGraph operations or it is too complicated. 
For example if you want to detect Convolution->Add sub-graph without specifying particular input type for Convolution operation or you want to create pattern where some of operations can have different types.
And for these cases nGraph provides additional helpers to construct patterns for GraphRewrite transformations. 

There are two main helpers:
1. `ngraph::pattern::any_input` - helps to express inputs if their types are undefined.
2. `ngraph::pattern::wrap_type<T>` - helps to express nodes of pattern without specifying node attributes.

Let's go through example to have better understanding how it works:

> **Note**: node attributes do not participate in pattern matching and needed only for operations creation. Only operation types participate in pattern matching.

Example below shows basic usage of `pattern::any_input`.
Here we construct Multiply pattern with arbitrary first input and Constant as a second input. 
Also as Multiply is commutative operation it does not matter in which order we set inputs (any_input/Constant or Constant/any_input) because both cases will be matched.

@snippet example_ngraph_utils.cpp pattern:label_example

This example show how we can construct pattern when operation has arbitrary number of inputs.

@snippet example_ngraph_utils.cpp pattern:concat_example

This example shows how to use predicate to construct pattern. Also it shows how to match pattern manually on given node.

@snippet example_ngraph_utils.cpp pattern:predicate_example

> **Note**: be careful with manual matching because Matcher object holds matched nodes. To clear match use m->clear_state() method.

## Working with ngraph::Function <a name="working_with_ngraph_function"></a>

In this chapter we will review nGraph API that allows us to manipulate with `ngraph::Function`.

###1. ngraph::Node input and output ports

First of all let's talk about `ngraph::Node` input/output ports. Each nGraph operation has input and output ports except cases when operation has `Result`, `Parameter` or `Constant` type.

Every port belongs to its node so using port we can access parent node, get shape and type for particular input/output, get all consumers in case of output port and get producer node in case of input port.
With output port we can set inputs for newly created operations. 

Lets look at code example.

@snippet example_ngraph_utils.cpp ngraph:ports_example

You may notice that we usually construct operations in this way:
```cpp
std::shared_ptr<Node> neg_const = opset1::Constant::create(sub->get_input_element_type(1), Shape{1}, {-1}));
Output<Node> data = node->input_value(0);
auto neg = std::make_shared<ngraph::opset1::Multiply>(data, neg_const);
```
In this example `opset3::Multiply` operation takes `Output<Node>` and `std::shared_ptr<Node>` as inputs. But constructor takes both as `Output<Node>`. 
In this case `std::shared_ptr<Node>` will be automatically converted to `Output<Node>` if node has exactly one output port otherwise conversion will raise an exception.   

###2. ngraph::Node replacement

nGraph provides two ways for node replacement: via nGraph helper function and directly via port methods. We are going to review both of them.

Let's start with nGraph helper functions. The most popular function is `ngraph::replace_node(old_node, new_node)`.

We will review real replacement case where Negative operation replaces with Multiply.

![ngraph_replace_node]

@snippet example_ngraph_utils.cpp ngraph:replace_node

`ngraph::replace_node` has a constraint that number of output ports for both of ops must be the same otherwise it will raise an exception.


The alternative way to do the same replacement is next:
```cpp
// All neg->output(0) consumers will be moved to mul->output(0) port
neg->output(0).replace(mul->output(0));
```

Another transformation example is insertion.

![ngraph_insert_node]

@snippet example_ngraph_utils.cpp ngraph:insert_node

The alternative way to insert operation is to make a node copy and use `replace_node`:

@snippet example_ngraph_utils.cpp ngraph:insert_node_with_copy

###3. ngraph::Node elimination

Another type of node replacement is its elimination.

To eliminate operation nGraph has special method that consider all limitations related to InferenceEngine.

@snippet example_ngraph_utils.cpp ngraph:eliminate_node

`replace_output_update_name` in case of successful replacement it automatically preserves friendly name and runtime info.
  

## Transformation writing essentials <a name="transformation_writing_essentials"></a>

When developing transformation we need to follow next transformation rules:

###1. Operation Set (OpSet)

Which OpSet to use in your transformation? The right answer is latest that exists at the moment. An exception is ConvertOpSetXToOpSetY transformations where operations from OpSetX and OpSetY are required to use.

@snippet example_ngraph_utils.cpp ngraph:include

###2. Dynamic Shape and Rank

nGraph has two types for shape representation: 
`ngraph::Shape` - represents static shape.
`ngraph::PartialShape` - represents dynamic shape. That means that rank or some of dimensions are dynamic (undefined).
`ngraph::PartialShape` can be converted to `ngraph::Shape` using `get_shape()` method if all dimensions are static otherwise conversion will raise an exception.

@snippet example_ngraph_utils.cpp ngraph:shape

But in most cases before getting static shape using `get_shape()` method you need to check that shape is static.  

Also if your transformation requires only input shape rank or particular dimension value for some reason please do not use `get_shape()` method. See example below how not to use `get_shape()`

@snippet example_ngraph_utils.cpp ngraph:shape_check

Not using `get_shape()` method makes your transformation more flexible and applicable for more cases.

###3. Friendly Names

Each `ngraph::Node` has unique name (is used for nGraph internals) and friendly name. In transformations we care only about friendly name because it represents name from IR. 
Also friendly name is used as output tensor name (until we do not have other way to represent output tensor name) and user code that requests intermediate outputs based on this names.
So not to loose friendly name when replacing node with other node or sub-graph we need to set original friendly name to the latest node in replacing sub-garph. See example below. 

```cpp
// Replace Div operation with Power and Multiply sub-graph and set original friendly name to Multiply operation
auto pow = std::make_shared<ngraph::opset1::Power>(div->input(1).get_source_output(),
                                                           op::Constant::create(div->get_input_element_type(1), Shape{1}, {-1}));
auto mul = std::make_shared<ngraph::opset1::Multiply>(div->input(0).get_source_output(), pow);
mul->set_friendly_name(div->get_friendly_name());
ngraph::replace_node(div, mul);
```

In more advanced cases when replaced operation has several outputs and we add additional consumers to its outputs we make decision how to set friendly name by arrangement.

###4. Runtime Info

Runtime info is a map `std::map<std::string, std::shared_ptr<Variant>>` located inside `ngraph::Node` class. It represents additional attributes in `ngraph::Node`.
These attributes can be set by users or by plugins and when executing transformation that changes `ngraph::Function` we need to preserve this attributes as they won't be automatically propagated.
In most cases transformations has next types: 1:1 (replace node with another node), 1:N (replace node with a sub-graph), N:1 (fuse sub-graph into a single node), N:M (any other transformation).
Currently there is no mechanism that automatically detects transformation types so we need to propagate this runtime information manually. See examples below.

```cpp
// Replace Transpose with Reshape operation (1:1)
ngraph::copy_runtime_info(transpose, reshape);
```

```cpp
// Replace Div operation with Power and Multiply sub-graph (1:N)
ngraph::copy_runtime_info(div, {pow, mul});
```

```cpp
// Fuse Convolution with Add operation (N:1)
ngraph::copy_runtime_info({conv, bias}, {conv_ie});
```

```cpp
// Any other transformation that replaces one sub-graph with another sub-graph (N:M)
ngraph::copy_runtime_info({a, b, c}, {e, f});
```

When transformation has multiple fusions or decompositions `ngraph::copy_runtime_info` must be called multiple times for each case. 

> **Note**: copy_runtime_info removes rt_info from destination nodes. If you want to keep it you need to specify them in source nodes like this: copy_runtime_info({a, b, c}, {a, b})

###5. Constant Folding

If your transformation inserts constant sub-graphs that needs to be folded do not forget to use `ngraph::pass::ConstantFolding()` after your transformation or call constant folding directly for operation.
Example below shows how constant sub-graph can be constructed.

```cpp
// After ConstantFolding pass Power will be replaced with Constant 
auto pow = std::make_shared<ngraph::opset3::Power>(
                    opset3::Constant::create(element::f32, Shape{1}, {2})
                    opset3::Constant::create(element::f32, Shape{1}, {3}));
auto mul = std::make_shared<ngraph::opset3::Multiply>(input /* not constant input */, pow);
``` 

Manual constant folding is more preferable than `ngraph::pass::ConstantFolding()` because it is much faster.

Below you can find an example of manual constant folding:

@snippet src/template_pattern_transformation.cpp manual_constant_folding

## Common mistakes in transformations <a name="common_mistakes"></a>

In transformation development process 

* Do not use deprecated nGraph API. Deprecated methods has `NGRAPH_DEPRECATED` macros in its definition. 
* Do not pass `shared_ptr<Node>` as input for other node if type of node is unknown or it has multiple outputs. Use explicit output port.
* If you replace node with another node that produce different shape you need to remember that new shape won't be propagated until first `validate_nodes_and_infer_types` call for `ngraph::Function`. If you are using `pass::Manager` it will automatically call this method after each transformation execution.
* Do not forget to call `ngraph::ConstantFolding` pass if your transformation creates constant sub-graphs.
* Use latest OpSet if you are not developing downgrade transformation pass.
* When developing callback for `ngraph::pass::MatcherPass` do not change nodes that comes after root node in topological order. 

## Using pass manager <a name="using_pass_manager"></a>

`ngraph::pass::Manager` is a container class that can store list of transformations and execute them. The main idea of this class is to have high-level representation for grouped list of transformations.
It can register and apply any [transformation types](#transformations_types) on function.
In addition `ngraph::pass::Manager` has extended debug capabilities (find more information in [how to debug transformations](#how_to_debug_transformations) section). 

Example below shows basic usage of `ngraph::pass::Manager`

@snippet src/template_pattern_transformation.cpp matcher_pass:manager3

Another example how multiple matcher passes can be united into single GraphRewrite.

@snippet src/template_pattern_transformation.cpp matcher_pass:manager2

## How to debug transformations <a name="how_to_debug_transformations"></a>

The most popular tool for transformations debugging is `ngraph::pass::VisualizeTree` transformation that visualize ngraph::Function.

Usage example:

@snippet example_ngraph_utils.cpp ngraph:visualize

`ngraph::pass::VisualizeTree` can be parametrized via environment variables:

```
NGRAPH_VISUALIZE_TREE_OUTPUT_SHAPES=1 - visualize shapes
NGRAPH_VISUALIZE_TREE_OUTPUT_TYPES=1  - visualize types
```

> **Note**: current VisualTree has not user friendly interface and it will be changed in nearest future. The intention is to move visualize abilities inside transformations.

If you are using `ngraph::pass::Manager` to run sequence of transformations you can get additional debug capabilities by using next environment variables:

```
NGRAPH_PROFILE_PASS_ENABLE=1 - enables performance measurement for each transformation and prints execution status
NGRAPH_ENABLE_VISUALIZE_TRACING=1 -  enables visualization after each transformation. By default it saves dot and svg files.
```

> **Note**: make sure that you have dot installed on your machine otherwise it will silently save only dot file without svg file.

## Disabling/Enabling specific transformations for plugin X	 <a name="disabling_transformation"></a>

This topic mostly related to conversion to legacy opset and plugins that based on CNNNetwork but still this mechanism can be applied for other cases.
Let's suppose that plugin X enabled `opset3::StridedSlice` operation support and you want to disable `ngraph::pass::ConvertStridedSliceToCrop` transformation for plugin X.
To do this you need to create callback on plugin side and pass it to transformation. And also you need to update particular transformation to use this callback.  

```cpp
// Update callback to be able to use m_transformation_callback if this transformation based on GraphRewrite.
ngraph::graph_rewrite_callback callback = [this](pattern::Matcher &m) {
    ...
}

// Use transformation_callback not to execute transformation if callback returns true for given node
if (m_transformation_callback(node)) {
    return false;
}

// Implement transformation callback and pass it directly to transformation or pass::Manager
const auto transformations_callback = [](const std::shared_ptr<const ::ngraph::Node> &node) -> bool {
    return std::dynamic_pointer_cast<const ::ngraph::opset3::StridedSlice>(node) != nullptr;
};

// Register transformation and pass callback to pass::Manager
ngraph::pass::Manager manager;
manager.register_pass<ngraph::pass::ConvertStridedSliceToCrop>();
// pass::Manager will set callback to all reistered transformations automatically
manager.set_callback(transformations_callback);
manager.run_passes(f);
```

## Transformations testing <a name="transformations_testing"></a>

If you are developing new transformation inside plugin you need to add test into `template_plugin/tests/functional/transformations` folder.
We have two types of tests: nGraph reader tests located in `inference-engine/tests/functional/inference_engine/ngraph_reader` and transformation tests located  in `inference-engine/tests/functional/inference_engine/transformations`
Reader tests are IR based and test end to end conversion from IR to CNNNetwork. Transformation tests test single ngraph transformations or low level functiont that are used inside transformations.

The basic transformation test looks like this:

@snippet tests/functional/transformations/template_transformations_test.cpp transformation:test


[ngraph_replace_node]: ../images/ngraph_replace_node.png
[ngraph_insert_node]: ../images/ngraph_insert_node.png
[transformations_structure]: ../images/transformations_structure.png
[register_new_node]: ../images/register_new_node.png
[graph_rewrite_execution]: ../images/graph_rewrite_execution.png
[graph_rewrite_efficient_search]: ../images/graph_rewrite_efficient_search.png