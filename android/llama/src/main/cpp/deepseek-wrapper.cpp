#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include "llama.h"
#include "common.h"

struct deepseek_context {
    struct llama_context * ctx_llama = nullptr;
    struct llama_model * model = nullptr;
};

struct deepseek_params {
    int32_t n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_batch = 512;
    std::string prompt;
    struct llama_sampling_params sampling;
};

struct deepseek_context * ctx_deepseek = nullptr;
struct deepseek_params params;
struct llama_model * model = nullptr;

void deepseek_init(const char * model_path, const char * type) {
    // Initialize llama.cpp parameters
    llama_backend_init();

    // Load the model
    llama_model_params model_params = llama_model_default_params();
    model = llama_load_model_from_file(model_path, model_params);
    if (model == nullptr) {
        fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, model_path);
        return;
    }
}

struct deepseek_context * deepseek_init_context(struct deepseek_params * params, struct llama_model * model) {
    struct deepseek_context * ctx = new deepseek_context;

    if (!ctx) {
        fprintf(stderr, "%s: error: failed to allocate deepseek context\n", __func__);
        return nullptr;
    }

    ctx->model = model;

    // Create the llama context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = params->n_batch;
    ctx_params.n_threads = params->n_threads;
    ctx_params.n_threads_batch = params->n_threads;

    ctx->ctx_llama = llama_new_context_with_model(model, ctx_params);
    if (!ctx->ctx_llama) {
        fprintf(stderr, "%s: error: failed to create the llama context with model\n", __func__);
        delete ctx;
        return nullptr;
    }

    return ctx;
}

void deepseek_free() {
    if (ctx_deepseek) {
        if (ctx_deepseek->ctx_llama) {
            llama_free(ctx_deepseek->ctx_llama);
            ctx_deepseek->ctx_llama = nullptr;
        }
        delete ctx_deepseek;
        ctx_deepseek = nullptr;
    }

    if (model) {
        llama_free_model(model);
        model = nullptr;
    }

    llama_backend_free();
}