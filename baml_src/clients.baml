// Retry policy for transient failures
retry_policy StandardRetry {
  max_retries 2
}

// --- Individual Providers ---

client<llm> OpenAI {
  provider openai
  retry_policy StandardRetry
  options {
    model "gpt-4o-mini"
    temperature 0
  }
}

client<llm> Bedrock {
  provider aws-bedrock
  retry_policy StandardRetry
  options {
    // Sonnet 4.5 cross-region inference profile (active; replaces retired Sonnet 4).
    // To move to Sonnet 5, confirm the Bedrock inference-profile ID and that model
    // access is enabled for it in your AWS account, then swap this one line.
    model "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    region env.AWS_REGION
    inference_configuration {
      temperature 0
      max_tokens 4096
    }
  }
}

client<llm> Gemini {
  provider google-ai
  retry_policy StandardRetry
  options {
    model "gemini-2.5-flash"
    api_key env.GEMINI_API_KEY
  }
}

// --- Fallback Chain ---
// Tries each provider in order until one succeeds

client<llm> Resilient {
  provider fallback
  options {
    strategy [
      Bedrock
      OpenAI
      Gemini
    ]
  }
}

// --- Local models ---
// Ollama's OpenAI-compatible endpoint. Opt-in only: never added to the
// Resilient fallback chain, because a 35B local model silently catching a
// Bedrock outage is not a failover anyone asked for. Select it explicitly
// with NAM_LLM_PROVIDER=ollama, which builds this client at runtime with
// NAM_OLLAMA_MODEL / NAM_OLLAMA_URL overrides.
//
// qwen-judge is a derived tag: the upstream coder build ships a red-team
// operator persona as its SYSTEM, which biases any judging task. The tag
// keeps the weights and replaces that prompt.
//
// max_tokens is the OUTPUT budget, not the context window — the model
// declares num_ctx 262144 itself. Thinking is on and thinking tokens count
// against this budget, the same trap providers.py documents for opus-5.
client<llm> QwenCoder {
  provider openai-generic
  retry_policy StandardRetry
  options {
    base_url "http://localhost:11434/v1"
    model "qwen-judge"
    api_key "ollama"
    temperature 0
    max_tokens 32768
  }
}
