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

struct deepseek_sampling_params {
    int32_t n_prev                = 64;     // number of previous tokens to remember
    int32_t n_probs              = 0;      // if greater than 0, output the probabilities of top n_probs tokens
    int32_t top_k                = 40;     // <= 0 to use vocab size
    float   top_p                = 0.95f;  // 1.0 = disabled
    float   min_p                = 0.05f;  // 0.0 = disabled
    float   tfs_z                = 1.00f;  // 1.0 = disabled
    float   typical_p            = 1.00f;  // 1.0 = disabled
    float   temp                 = 0.80f;  // 1.0 = disabled
    float   penalty_last_n       = 64;     // last n tokens to penalize
    float   penalty_repeat       = 1.10f;  // 1.0 = disabled
    float   penalty_freq         = 0.00f;  // 0.0 = disabled
    float   penalty_present      = 0.00f;  // 0.0 = disabled
    int32_t mirostat            = 0;      // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float   mirostat_tau        = 5.00f;  // target entropy
    float   mirostat_eta        = 0.10f;  // learning rate
    bool    penalize_nl         = true;   // consider newlines as a repeatable token
};

struct deepseek_context {
    struct llama_context * ctx_llama = nullptr;
    struct llama_model * model = nullptr;
};

struct deepseek_params {
    int32_t n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_batch = 512;
    std::string prompt;
    struct deepseek_sampling_params sampling;
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