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
    model "us.anthropic.claude-sonnet-4-20250514-v1:0"
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
