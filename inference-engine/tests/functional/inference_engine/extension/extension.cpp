// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_core.hpp>
#include <ngraph/ngraph.hpp>
#include <onnx_import/onnx_utils.hpp>

#include "extension.hpp"
#include "op.hpp"


class CustomReluKernel : public InferenceEngine::ILayerExecImpl {
    public:
        explicit CustomReluKernel(const std::shared_ptr<ngraph::Node>& node): node(node) {}

        InferenceEngine::StatusCode
        init(InferenceEngine::LayerConfig& /*config*/, InferenceEngine::ResponseDesc* /*resp*/) noexcept override {
            return InferenceEngine::StatusCode::OK;
        }

        InferenceEngine::StatusCode getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig>& conf,
                                                               InferenceEngine::ResponseDesc* /*resp*/) noexcept override {
            InferenceEngine::LayerConfig layerConfig;
            layerConfig.dynBatchSupport = true;

            if (node->outputs().size() != 1 && node->inputs().size() != 1)
                return InferenceEngine::GENERAL_ERROR;

            InferenceEngine::DataConfig cfg;
            cfg.constant = false;
            cfg.inPlace = 0;

            InferenceEngine::SizeVector order;
            auto partialShape = node->get_output_partial_shape(0);
            if (partialShape.is_dynamic())
                return InferenceEngine::GENERAL_ERROR;

            auto shape = node->get_output_shape(0);
            for (size_t i = 0; i < shape.size(); i++) {
                order.push_back(i);
            }
            cfg.desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32,
                                                   shape, {shape, order});
            layerConfig.outConfs.push_back(cfg);
            layerConfig.inConfs.push_back(cfg);
            conf.push_back(layerConfig);
            return InferenceEngine::OK;
        }

        InferenceEngine::StatusCode
        execute(std::vector<InferenceEngine::Blob::Ptr>& inputs, std::vector<InferenceEngine::Blob::Ptr>& outputs,
                InferenceEngine::ResponseDesc* /*resp*/) noexcept override {
            for (size_t i = 0; i < inputs.size(); i++) {
                InferenceEngine::MemoryBlob::CPtr minput = InferenceEngine::as<InferenceEngine::MemoryBlob>(inputs[i]);
                InferenceEngine::MemoryBlob::Ptr moutput = InferenceEngine::as<InferenceEngine::MemoryBlob>(outputs[i]);
                if (!moutput || !minput) {
                    return InferenceEngine::StatusCode::PARAMETER_MISMATCH;
                }
                // locked memory holder should be alive all time while access to its buffer happens
                auto minputHolder = minput->rmap();
                auto moutputHolder = moutput->wmap();

                auto inputData = minputHolder.as<const float *>();
                auto outputData = moutputHolder.as<float  *>();
                for (size_t j = 0; j < minput->size(); j++) {
                    outputData[j] = inputData[j] < 0 ? inputData[j] * 2 : inputData[j];
                }
            }
            return InferenceEngine::StatusCode::OK;
        }



    private:
        const std::shared_ptr<ngraph::Node> node;
};

class CustomRelu : public ngraph::op::Op {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"CustomRelu", 100500};
    const ngraph::NodeTypeInfo& get_type_info() const override { return type_info;  }
    CustomRelu() = default;
    CustomRelu(const ngraph::Output<ngraph::Node>& arg): ngraph::op::Op({arg}) {
        constructor_validate_and_infer_types();
    }
    void validate_and_infer_types() override {
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override {
        return std::make_shared<CustomRelu>(new_args.at(0));
    }
    bool visit_attributes(ngraph::AttributeVisitor& visitor) override {
        return true;
    }
};

constexpr ngraph::NodeTypeInfo CustomRelu::type_info;

class CustomReluExtension : public InferenceEngine::IExtension {
    public:
        CustomReluExtension() {
        }

        void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept override {}

        void Release() noexcept override { delete this; }

        void Unload() noexcept override {}

        std::map<std::string, ngraph::OpSet> getOpSets() override {
            std::map<std::string, ngraph::OpSet> opsets;
            ngraph::OpSet opset;
            opset.insert<CustomRelu>();
            opsets["custom_opset"] = opset;
            return opsets;
        }

        std::vector<std::string> getImplTypes(const std::shared_ptr<ngraph::Node>& node) override {
            if (node->description() != CustomRelu::type_info.name)
                return {};
            return {"CPU"};
        }

        InferenceEngine::ILayerImpl::Ptr getImplementation(const std::shared_ptr<ngraph::Node>& node, const std::string& implType) override {
            return std::make_shared<CustomReluKernel>(node);
        }
};

template <typename Extension, typename CustomOp>
void test_extension(const std::string& model, const std::vector<float>& input_values, const std::vector<float>& expected,
                    bool register_onnx_op, int opset_ver = 0, std::string domain = "") {
    InferenceEngine::Core ie;
    InferenceEngine::Blob::CPtr weights;
    ie.AddExtension(std::make_shared<Extension>());
    if (register_onnx_op) {
        ngraph::onnx_import::register_operator(
            CustomOp::type_info.name, opset_ver, domain, [](const ngraph::onnx_import::Node& node) -> ngraph::OutputVector {
                ngraph::OutputVector ng_inputs{node.get_ng_inputs()};
                return {std::make_shared<CustomRelu>(ng_inputs.at(0))};
        });
    }
    auto network = ie.ReadNetwork(model, weights);
    auto function = network.getFunction();

    int ops_count = 0;
    for (auto op : function->get_ops()) {
        const auto op_type = std::string(op->get_type_name());
        ops_count += (op_type == CustomOp::type_info.name ? 1 : 0);
    }
    ASSERT_EQ(ops_count, 1);

    auto network_inputs = network.getInputsInfo();
    auto network_outputs = network.getOutputsInfo();
    auto exe_network = ie.LoadNetwork(network, "CPU");
    auto inference_req = exe_network.CreateInferRequest();
    const auto& input = network_inputs.begin();
    const auto& input_info = input->second;

    auto blob = std::make_shared<InferenceEngine::TBlob<float>>(input_info->getTensorDesc());
    blob->allocate();
    ASSERT_EQ(input_values.size(), blob->size());
    float* blob_buffer = blob->wmap().template as<float*>();
    std::copy(input_values.begin(), input_values.end(), blob_buffer);
    inference_req.SetBlob(input->first, blob);

    inference_req.Infer();

    auto output = network_outputs.begin();
    InferenceEngine::MemoryBlob::CPtr computed = InferenceEngine::as<InferenceEngine::MemoryBlob>(inference_req.GetBlob(output->first));
    const auto computed_data = computed->rmap();
    const auto* computed_data_buffer = computed_data.template as<const float*>();
    std::vector<float> computed_values(computed_data_buffer,
                                   computed_data_buffer + computed->size());
    ASSERT_EQ(expected, computed_values);
}


TEST(Extension, OnnxModelWithCustomRelu) {
    std::string model = R"V0G0N(
ir_version: 3
producer_name: "nGraph ONNX Importer"
graph {
  node {
    input: "A"
    output: "Y"
    name: "customrelu"
    op_type: "CustomRelu"
    domain: "custom_domain"
  }
  name: "test_graph"
  input {
    name: "A"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 10
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 10
          }
        }
      }
    }
  }
}
opset_import {
  version: 4
}
)V0G0N";

    std::vector<float> input_values{1, -2, 3, -4, 5, -6, 7, -8, 9, -10};
    std::vector<float> expected{1, -4, 3, -8, 5, -12, 7, -16, 9, -20};
    test_extension<CustomReluExtension, CustomRelu>(model, input_values, expected, true, 1, "custom_domain");
}


TEST(Extension, XmlModelWithCustomRelu) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="10"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="CustomRelu" version="custom_opset">
            <input>
                <port id="1" precision="FP32">
                    <dim>10</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>10</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";

    std::vector<float> input_values{1, -2, 3, -4, 5, -6, 7, -8, 9, -10};
    std::vector<float> expected{1, -4, 3, -8, 5, -12, 7, -16, 9, -20};
    test_extension<CustomReluExtension, CustomRelu>(model, input_values, expected, false);
}


TEST(Extension, XmlModelWithExtensionFromDSO) {
    std::string model = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="2,2,2,1"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer name="operation" id="1" type="Template" version="custom_opset">
            <data  add="11"/>
            <input>
                <port id="1" precision="FP32">
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";

    std::vector<float> input_values{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> expected{12, 13, 14, 15, 16, 17, 18, 19};
    test_extension<TemplateExtension::Extension, TemplateExtension::Operation>(model, input_values, expected, false);
}


TEST(Extension, OnnxModelWithExtensionFromDSO) {
    std::string model = R"V0G0N(
ir_version: 3
producer_name: "nGraph ONNX Importer"
graph {
  node {
    input: "A"
    output: "Y"
    name: "operation"
    op_type: "Template"
    domain: "custom_domain"
    attribute {
        name: "add"
        type: INT
        i: 11
    }
  }
  name: "test_graph"
  input {
    name: "A"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 2
          }
          dim {
            dim_value: 2
          }
          dim {
            dim_value: 2
          }
          dim {
            dim_value: 1
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 2
          }
          dim {
            dim_value: 2
          }
          dim {
            dim_value: 2
          }
          dim {
            dim_value: 1
          }
        }
      }
    }
  }
}
opset_import {
  version: 4
}
)V0G0N";

    std::vector<float> input_values{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> expected{12, 13, 14, 15, 16, 17, 18, 19};
    test_extension<TemplateExtension::Extension, TemplateExtension::Operation>(model, input_values, expected, false);
}


