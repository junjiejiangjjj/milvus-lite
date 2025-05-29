// Copyright (C) 2019-2024 Zilliz. All rights reserved.
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

#include <memory>
#include "pb/schema.pb.h"
#include "pb/common.pb.h"
#include "pb/milvus.pb.h"
#include "status.h"

namespace milvus::local::function {

using milvus::local::Status;

class FunctionBase {
 public:
    FunctionBase(const milvus::proto::schema::CollectionSchema* schema,
                 const milvus::proto::schema::FunctionSchema* func_schema)
        : schema_(schema), func_schema_(func_schema) {
        collection_name_ = schema->collection_name();
        function_type_name_ = func_schema->type();
        function_name_ = func_schema->name();
    }

    virtual ~FunctionBase() = default;

    const milvus::proto::schema::FunctionSchema*
    GetSchema() const {
        return schema_;
    }

    const std::vector<milvus::proto::schema::FieldSchema*>
    GetOutputFields() const {
        return output_fields_;
    }

 public:
    virtual Status
    ProcessInsert(const std::vector<milvus::proto::schema::FieldData*>& inputs,
                  std::vector<milvus::proto::schema::FieldData*>* outputs) = 0;
    virtual Status
    ProcessSearch(
        const milvus::proto::common::PlaceholderValue& placeholder_value,
        std::vector<milvus::proto::schema::FieldData*>* outputs) = 0;

 private:
    const milvus::proto::schema::FunctionSchema* schema_;
    const std::vector<milvus::proto::schema::FieldSchema*> output_fields_;

    const std::string collection_name_;
    const std::string function_type_name_;
    const std::string function_name_;
};

}  // namespace milvus::local::function
