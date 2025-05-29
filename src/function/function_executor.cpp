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

#include "function/function_executor.h"
#include <memory>
#include <utility>
#include "common.h"
#include "function/bm25_function.h"
#include "function/function_util.h"
#include "schema.pb.h"
#include "status.h"

namespace milvus::local::function {

std::pair<Status, std::unique_ptr<FunctionExecutor>>
FunctionExecutor::Create(
    milvus::proto::schema::CollectionSchema* schema,
    const ::milvus::proto::schema::FieldSchema& ann_field) {
    for (const auto& f_schemn : schema->functions()) {
        if (f_schemn.output_field_names(0) == ann_field.name()) {
            auto [s, f] = CreateFunction(schema, &f_schemn);
            if (!s.IsOk()) {
                return std::make_pair(s, nullptr);
            }

            std::unique_ptr<FunctionExecutor> executor(
                new FunctionExecutor(std::move(f)));
            return std::make_pair(Status::Ok(), std::move(executor));
        }
    }
    return std::make_pair(Status::ParameterInvalid("No function's output is {}",
                                                   ann_field.name()),
                          nullptr);
}

std::pair<Status, std::unique_ptr<TransformFunctionBase>>
CreateFunction(const milvus::proto::schema::CollectionSchema* schema,
               const milvus::proto::schema::FunctionSchema* function_schema) {
    if (function_schema->type() == milvus::proto::schema::FunctionType::BM25) {
    } else {
        return std::make_pair(
            Status::ParameterInvalid("Unsupported function: {}",
                                     milvus::proto::schema::FunctionType_Name(
                                         function_schema->type())),
            nullptr);
    }
    auto [s, f] = BM25Function::NewBM25Function(schema, function_schema);
    if (!s.IsOk()) {
        return std::make_pair(s, nullptr);
    }
    return std::make_pair(Status::Ok(), std::move(f));
}

Status
FunctionExecutor::ProcessInsert(milvus::proto::milvus::InsertRequest* insert) {
    return Status::Ok();
}

Status
FunctionExecutor::ProcessSearch(milvus::proto::milvus::SearchRequest* search) {
    milvus::proto::common::PlaceholderGroup ph_group;
    if (!ph_group.ParseFromString(search->placeholder_group())) {
        return Status::ParameterInvalid("Parse placehoder string failed");
    }

    if (ph_group.placeholders_size() != 1) {
        return Status::ParameterInvalid("placeholders size is not equal 1");
    }

    milvus::proto::common::PlaceholderGroup output;
    CHECK_STATUS(function_->ProcessSearch(ph_group, &output), "");
    search->mutable_placeholder_group()->assign(output.SerializeAsString());
    return Status::Ok();
}

}  // namespace milvus::local::function
