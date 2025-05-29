// Copyright (C) 2019-2025 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#pragma once

#include "function/function.h"
#include "tantivy/tokenizer.h"

namespace milvus::local::function {

class BM25Function : public FunctionBase {
 public:
    BM25Function(
        const milvus::proto::schema::FunctionSchema* schema,
        const std::vector<milvus::proto::schema::FieldSchema*> output_fields,
        const std::string collection_name,
        const std::string function_type_name,
        const std::string function_name,
        const std::string provider)
        : FunctionBase(schema,
                       output_fields,
                       collection_name,
                       function_type_name,
                       function_name,
                       provider) {
    }

 public:
    Status
    ProcessInsert(
        const std::vector<milvus::proto::schema::FieldData*>& inputs,
        std::vector<milvus::proto::schema::FieldData*>* outputs) override;
    Status
    ProcessSearch(
        const milvus::proto::common::PlaceholderValue& placeholder_value,
        std::vector<milvus::proto::schema::FieldData*>* outputs) override;

 private:
    milvus::tantivy::Tokenizer tokenizer_;
};

}  // namespace milvus::local::function