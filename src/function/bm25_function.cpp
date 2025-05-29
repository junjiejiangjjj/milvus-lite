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

#include "function/bm25_function.h"

namespace milvus::local::function {

Status
BM25Function::ProcessInsert(
    const std::vector<milvus::proto::schema::FieldData*>& inputs,
    std::vector<milvus::proto::schema::FieldData*>* outputs) {
    if (inputs.size() != 1) {
        return Status::ParameterInvalid("BM25Function inputs size must be 1");
    }

    if (inputs[0]->type() != milvus::proto::schema::DataType::VarChar) {
        return Status::ParameterInvalid(
            "BM25Function input type must be VarChar");
    }

    if (!inputs[0]->scalars().has_string_data()) {
        return Status::ParameterInvalid(
            "BM25Function input data is not string");
    }

    auto doc = inputs[0]->scalars().string_data();

    return Status::Ok();
}

Status
BM25Function::ProcessSearch(
    const milvus::proto::common::PlaceholderValue& placeholder_value,
    std::vector<milvus::proto::schema::FieldData*>* outputs) {
    return Status::Ok();
}

}  // namespace milvus::local::function