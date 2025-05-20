#include <android/log.h>
#include <jni.h>
#include <iomanip>
#include <math.h>
#include <string>
#include <unistd.h>
#include "llama.h"
#include "deepseek-wrapper.cpp"
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

extern "C" JNIEXPORT void JNICALL
Java_com_nexa_NexaDeepSeekInference_init(JNIEnv *env, jobject /* this */, jstring jmodel, jstring jtype) {
    const char* model_chars = env->GetStringUTFChars(jmodel, nullptr);
    const char* type = env->GetStringUTFChars(jtype, nullptr);

    deepseek_init(model_chars, type);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_nexa_NexaDeepSeekInference_sampler_1init(JNIEnv *env, jobject /* this */, jstring jprompt, jlong jnpast) {
    const char* prompt = env->GetStringUTFChars(jprompt, nullptr);
    auto* n_past = reinterpret_cast<int*>(jnpast);

    ctx_deepseek = deepseek_init_context(&params, model);
    params.prompt = prompt;
    
    // Format the prompt for DeepSeek model
    params.prompt = "<|im_start|>system\nYou are DeepSeek-R1, created by Nexa AI. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + params.prompt + "<|im_end|>\n<|im_start|>assistant\n";
    
    params.sampling.top_k = 1;
    params.sampling.top_p = 1.0f;
    eval_string(ctx_deepseek->ctx_llama, params.prompt.c_str(), params.n_batch, n_past, true);

    struct common_sampler * smpl = common_sampler_init(ctx_deepseek->model, params.sampling);

    return reinterpret_cast<jlong>(smpl);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_nexa_NexaDeepSeekInference_npast_1init(JNIEnv *env, jobject /* this */) {
    int* n_past = new int(0);
    return reinterpret_cast<jlong>(n_past);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_nexa_NexaDeepSeekInference_inference(JNIEnv *env, jobject /* this */, jlong jnpast, jlong jsampler) {
    auto* n_past = reinterpret_cast<int*>(jnpast);
    auto * sampler =  reinterpret_cast<struct common_sampler *>(jsampler);
    const char * tmp = sample(sampler, ctx_deepseek->ctx_llama, n_past);

    jstring new_token = nullptr;
    new_token = env->NewStringUTF(tmp);
    return new_token;
}

extern "C" JNIEXPORT void JNICALL
Java_com_nexa_NexaDeepSeekInference_sampler_1free(JNIEnv *env, jobject /* this */, jlong jsampler) {
    struct common_sampler * sampler =  reinterpret_cast<struct common_sampler *>(jsampler);
    common_sampler_free(sampler);
}

extern "C" JNIEXPORT void JNICALL
Java_com_nexa_NexaDeepSeekInference_free(JNIEnv *env, jobject /* this */) {
    deepseek_free();
}