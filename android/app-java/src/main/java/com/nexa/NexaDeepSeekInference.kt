package com.nexa

import android.util.Log
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

/**
 * Mock implementation of NexaDeepSeekInference for building without native libraries
 */
class NexaDeepSeekInference(
    private val modelPath: String,
    private val stopWords: List<String> = listOf("</s>"),
    private val temperature: Float = 0.8f,
    private val maxNewTokens: Int = 512,
    private val topK: Int = 40,
    private val topP: Float = 0.9f
) {
    private val TAG = "NexaDeepSeekInference"
    
    fun loadModel() {
        Log.d(TAG, "Mock: Loading DeepSeek model from $modelPath")
    }

    fun createCompletionStream(
        prompt: String,
        stopWords: List<String> = this.stopWords,
        temperature: Float = this.temperature,
        maxNewTokens: Int = this.maxNewTokens,
        topK: Int = this.topK,
        topP: Float = this.topP
    ): Flow<String> = flow {
        Log.d(TAG, "Mock: Creating completion stream with prompt: $prompt")
        
        // Emit a mock response
        val mockResponse = "This is a mock response from DeepSeek model. The actual model is not loaded."
        for (word in mockResponse.split(" ")) {
            emit("$word ")
            kotlinx.coroutines.delay(200) // Simulate delay between tokens
        }
    }

    fun dispose() {
        Log.d(TAG, "Mock: Disposing DeepSeek model")
    }
}