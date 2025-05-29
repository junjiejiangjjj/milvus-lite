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

#include <map>
#include <cstddef>

#include "function/function.h"
#include "status.h"

namespace milvus::local::function {

using milvus::local::Status;

class FunctionExecutor {
 public:
    FunctionExecutor(milvus::proto::schema::CollectionSchema* schema,
                     milvus::proto::schema::FunctionSchema* function_schema);
    virtual ~FunctionExecutor() = default;

 public:
    Status
    ProcessInsert(milvus::proto::milvus::InsertRequest* insert);

    Status
    ProcessSearch(milvus::proto::milvus::SearchRequest* search);

 private:
    Status
    CreateFunction(milvus::proto::schema::CollectionSchema* schema,
                   milvus::proto::schema::FunctionSchema* function_schema);

 private:
    std::map<int64_t, FunctionBase*> functions_;
};

}  // namespace milvus::local::function