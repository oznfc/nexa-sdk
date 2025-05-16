#include <android/log.h>
#include <jni.h>
#include <iomanip>
#include <math.h>
#include <string>
#include <unistd.h>
#include "llama.h"
#include <nlohmann/json.hpp>
#include <jni.h>
#include <string>
#include <iostream>
#include <thread>

#define TAG "deepseek-android.cpp"
#define LOGi(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGe(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

extern bool is_valid_utf8(const char* str);
extern std::string jstring2str(JNIEnv* env, jstring jstr);

// Global variables
struct llama_context * ctx_llama = nullptr;
struct llama_model * model = nullptr;
struct llama_sampling_params sparams;

// Initialize the model
void deepseek_init(const char* model_path, const char* type) {
    LOGi("Initializing DeepSeek model: %s", model_path);
    
    // Initialize parameters
    llama_backend_init();
    
    // Model parameters
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99; // Use GPU if available
    
    // Context parameters
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = 512;
    
    // Load the model
    model = llama_model_load_from_file(model_path, model_params);
    if (model == nullptr) {
        LOGe("Failed to load model: %s", model_path);
        return;
    }
    
    // Create context
    ctx_llama = llama_new_context_with_model(model, ctx_params);
    if (ctx_llama == nullptr) {
        LOGe("Failed to create context");
        llama_model_free(model);
        model = nullptr;
        return;
    }
    
    // Initialize sampling parameters
    sparams = llama_sampling_default_params();
    sparams.temp = 0.8f;
    sparams.top_k = 40;
    sparams.top_p = 0.95f;
    sparams.repeat_penalty = 1.1f;
    
    LOGi("DeepSeek model initialized successfully");
}

// Free resources
void deepseek_free() {
    if (ctx_llama != nullptr) {
        llama_free(ctx_llama);
        ctx_llama = nullptr;
    }
    
    if (model != nullptr) {
        llama_model_free(model);
        model = nullptr;
    }
    
    llama_backend_free();
}

// Evaluate a string and update n_past
void eval_string(struct llama_context * ctx, const char * str, int n_batch, int * n_past, bool add_bos) {
    std::vector<llama_token> embd_inp;
    
    if (add_bos) {
        embd_inp.push_back(llama_token_bos(llama_model_get_vocab(model)));
    }
    
    // Tokenize the input string
    const int n_ctx = llama_n_ctx(ctx);
    std::vector<llama_token> tokens(n_ctx);
    int n_tokens = llama_tokenize(llama_model_get_vocab(model), str, strlen(str), tokens.data(), tokens.size(), true, false);
    tokens.resize(n_tokens);
    
    embd_inp.insert(embd_inp.end(), tokens.begin(), tokens.end());
    
    if (embd_inp.size() > n_ctx) {
        LOGe("Input is too long (%d tokens, max %d)", (int)embd_inp.size(), n_ctx);
        return;
    }
    
    // Process the tokens in batches
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    
    for (size_t i = 0; i < embd_inp.size(); i += n_batch) {
        size_t n_eval = std::min(n_batch, (int)(embd_inp.size() - i));
        
        batch.n_tokens = n_eval;
        for (size_t j = 0; j < n_eval; j++) {
            batch.token[j] = embd_inp[i + j];
            batch.pos[j] = *n_past + j;
            batch.seq_id[j][0] = 0;
            batch.n_seq_id[j] = 1;
        }
        
        if (llama_decode(ctx, batch) != 0) {
            LOGe("Failed to decode");
            llama_batch_free(batch);
            return;
        }
        
        *n_past += n_eval;
    }
    
    llama_batch_free(batch);
}

// Sampler structure
struct common_sampler {
    struct llama_sampling_context * ctx_sampling;
};

// Initialize sampler
struct common_sampler * common_sampler_init(struct llama_model * model, struct llama_sampling_params params) {
    struct common_sampler * sampler = new common_sampler();
    sampler->ctx_sampling = llama_sampling_init(params);
    return sampler;
}

// Free sampler
void common_sampler_free(struct common_sampler * sampler) {
    if (sampler != nullptr) {
        if (sampler->ctx_sampling != nullptr) {
            llama_sampling_free(sampler->ctx_sampling);
        }
        delete sampler;
    }
}

// Sample a token
const char * sample(struct common_sampler * sampler, struct llama_context * ctx, int * n_past) {
    llama_token id = 0;
    
    // Sample a token
    id = llama_sampling_sample(sampler->ctx_sampling, ctx, NULL);
    
    // If end of stream, return empty string
    if (id == llama_token_eos(llama_model_get_vocab(model))) {
        return "";
    }
    
    // Evaluate the token
    llama_batch batch = llama_batch_init(1, 0, 1);
    batch.token[0] = id;
    batch.pos[0] = *n_past;
    batch.n_tokens = 1;
    batch.seq_id[0][0] = 0;
    batch.n_seq_id[0] = 1;
    
    if (llama_decode(ctx, batch) != 0) {
        LOGe("Failed to decode");
        llama_batch_free(batch);
        return "";
    }
    
    llama_batch_free(batch);
    (*n_past)++;
    
    // Convert token to string
    static std::string result;
    result.resize(32);
    
    const int n_token_chars = llama_token_to_piece(llama_model_get_vocab(model), id, result.data(), result.size(), false, false);
    if (n_token_chars < 0) {
        result.resize(-n_token_chars);
        llama_token_to_piece(llama_model_get_vocab(model), id, result.data(), result.size(), false, false);
    } else {
        result.resize(n_token_chars);
    }
    
    return result.c_str();
}

// JNI functions
extern "C" JNIEXPORT void JNICALL
Java_com_nexa_NexaDeepSeekInference_init(JNIEnv *env, jobject /* this */, jstring jmodel, jstring jtype) {
    const char* model_chars = env->GetStringUTFChars(jmodel, nullptr);
    const char* type = env->GetStringUTFChars(jtype, nullptr);
    
    deepseek_init(model_chars, type);
    
    env->ReleaseStringUTFChars(jmodel, model_chars);
    env->ReleaseStringUTFChars(jtype, type);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_nexa_NexaDeepSeekInference_npast_1init(JNIEnv *env, jobject /* this */) {
    int* n_past = new int(0);
    return reinterpret_cast<jlong>(n_past);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_nexa_NexaDeepSeekInference_sampler_1init(JNIEnv *env, jobject /* this */, jstring jprompt, jlong jnpast) {
    auto* n_past = reinterpret_cast<int*>(jnpast);
    const char* prompt = env->GetStringUTFChars(jprompt, nullptr);
    
    // Format the prompt for DeepSeek model
    std::string formatted_prompt = "<|im_start|>system\nYou are DeepSeek-R1, created by Nexa AI. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n";
    formatted_prompt += prompt;
    formatted_prompt += "<|im_end|>\n<|im_start|>assistant\n";
    
    // Evaluate the prompt
    eval_string(ctx_llama, formatted_prompt.c_str(), 512, n_past, true);
    
    // Initialize sampler
    struct common_sampler * sampler = common_sampler_init(model, sparams);
    
    env->ReleaseStringUTFChars(jprompt, prompt);
    
    return reinterpret_cast<jlong>(sampler);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_nexa_NexaDeepSeekInference_inference(JNIEnv *env, jobject /* this */, jlong jnpast, jlong jsampler) {
    auto* n_past = reinterpret_cast<int*>(jnpast);
    auto* sampler = reinterpret_cast<struct common_sampler *>(jsampler);
    
    const char* token = sample(sampler, ctx_llama, n_past);
    
    jstring new_token = nullptr;
    new_token = env->NewStringUTF(token);
    return new_token;
}

extern "C" JNIEXPORT void JNICALL
Java_com_nexa_NexaDeepSeekInference_sampler_1free(JNIEnv *env, jobject /* this */, jlong jsampler) {
    struct common_sampler* sampler = reinterpret_cast<struct common_sampler*>(jsampler);
    common_sampler_free(sampler);
}

extern "C" JNIEXPORT void JNICALL
Java_com_nexa_NexaDeepSeekInference_free(JNIEnv *env, jobject /* this */) {
    deepseek_free();
}