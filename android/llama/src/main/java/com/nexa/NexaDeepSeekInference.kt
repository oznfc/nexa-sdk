package com.nexa
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn

class NexaDeepSeekInference(
    private val modelPath: String,
    private var stopWords: List<String> = emptyList(),
    private var temperature: Float = 0.8f,
    private var maxNewTokens: Int = 64,
    private var topK: Int = 40,
    private var topP: Float = 0.95f
) {
    init {
        System.loadLibrary("deepseek-android")
    }
    private var sampler_pointer: Long = 0
    private var nPastPointer: Long = 0
    private var generatedTokenNum: Int = 0
    private var generatedText: String = ""
    private var isModelLoaded: Boolean = false
    private var type:String = "deepseek-r1-distill"

    private external fun init(model: String, type: String)
    private external fun sampler_free(sampler:Long)
    private external fun free()

    private external fun sampler_init(prompt: String, npast: Long): Long
    private external fun inference(npast: Long, sampler:Long): String
    private external fun npast_init():Long

    @Synchronized
    fun loadModel() {
        if(isModelLoaded){
            throw RuntimeException("Model is already loaded.")
        }
        try {
            init(modelPath, type)
            isModelLoaded = true
        } catch (e: Exception) {
            println(e)
        } catch (e: UnsatisfiedLinkError) {
            throw RuntimeException("Native method not found: ${e.message}")
        }
    }

    fun dispose() {
        free()
    }

    private fun updateParams(
        stopWords: List<String>? = null,
        temperature: Float? = null,
        maxNewTokens: Int? = null,
        topK: Int? = null,
        topP: Float? = null
    ) {
        if(stopWords != null){
            this.stopWords = stopWords
        }
        if (temperature != null) {
            this.temperature = temperature
        }
        if (maxNewTokens != null) {
            this.maxNewTokens = maxNewTokens
        }
        if (topK != null) {
            this.topK = topK;
        }
        if (topP != null) {
            this.topP = topP
        }
    }

    private fun shouldStop(): Boolean {
        if(this.generatedTokenNum >= this.maxNewTokens){
            return true
        }

        return stopWords.any { generatedText.contains(it, ignoreCase = true) }
    }

    private fun resetGeneration() {
        generatedTokenNum = 0
        generatedText = ""
    }

    @Synchronized
    fun createCompletionStream(
        prompt: String,
        stopWords: List<String>? = null,
        temperature: Float? = null,
        maxNewTokens: Int? = null,
        topK: Int? = null,
        topP: Float? = null
    ): Flow<String> = flow {
        if(!isModelLoaded){
            throw RuntimeException("Model is not loaded.")
        }

        // Reset generation state at the start
        resetGeneration()
        updateParams(stopWords, temperature, maxNewTokens, topK, topP)
        nPastPointer = npast_init();
        sampler_pointer = sampler_init(prompt, nPastPointer)

        try {
            while (true) {
                val sampledText = inference(nPastPointer, sampler_pointer)
                generatedTokenNum += 1
                generatedText += sampledText
                if(shouldStop()){
                    break
                }
                emit(sampledText)
            }
        } finally {
            // Clean up resources and reset generation state
            resetGeneration()
            sampler_free(sampler_pointer)
        }
    }.flowOn(Dispatchers.IO)
}