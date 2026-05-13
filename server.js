// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = false; // Set to true to enable chat_template_kwargs thinking parameter

// Model mapping (adjust based on available NIM models)
const MODEL_MAPPING = {
  'deepseek': 'deepseek-ai/deepseek-v4-pro',
  'deepseek-flash': 'deepseek-ai/deepseek-v4-flash'
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    // Smart model selection with fallback
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
  const modelLower = model.toLowerCase();

  if (
    modelLower.includes('gpt-4') ||
    modelLower.includes('claude-opus') ||
    modelLower.includes('405b')
  ) {
    nimModel = 'meta/llama-3.1-405b-instruct';

  } else if (
    modelLower.includes('claude') ||
    modelLower.includes('gemini') ||
    modelLower.includes('70b')
  ) {
    nimModel = 'meta/llama-3.1-70b-instruct';

  } else {
    nimModel = 'meta/llama-3.1-8b-instruct';
  }
}
    
    // Transform OpenAI request to NIM format
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 9024,
      extra_body: ENABLE_THINKING_MODE ? { chat_template_kwargs: { thinking: true } } : undefined,
      stream: stream || false
    };
    
    // Make request to NVIDIA NIM API
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });
    
    if (stream) {
      // Handle streaming response with reasoning
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n');
              return;
            }
            
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content;
                
                if (SHOW_REASONING) {
                  let combinedContent = '';
                  
                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }
                  
                  if (content && reasoningStarted) {
                    combinedContent += '</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }
                  
                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                    delete data.choices[0].delta.reasoning_content;
                  }
                } else {
                  if (content) {
                    data.choices[0].delta.content = content;
                  } else {
                    data.choices[0].delta.content = '';
                  }
                  delete data.choices[0].delta.reasoning_content;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n');
            }
          }
        });
      });
      
      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });
    } else {
      // Transform NIM response to OpenAI format with reasoning
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }
          
          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
  console.error('Proxy error:', error.message);

  const status = error.response?.status || 500;
  const data = error.response?.data || {};
  const headers = error.response?.headers || {};

  // Enhanced 429 handling
  if (status === 429) {
    const retryAfter =
      headers['retry-after'] ||
      headers['x-ratelimit-reset'] ||
      null;

    const remainingRequests =
      headers['x-ratelimit-remaining-requests'];

    const remainingTokens =
      headers['x-ratelimit-remaining-tokens'];

    const limitRequests =
      headers['x-ratelimit-limit-requests'];

    const limitTokens =
      headers['x-ratelimit-limit-tokens'];

    const requestId =
      headers['x-request-id'] ||
      headers['request-id'];

    // Try to infer the reason
    let rateLimitType = 'rate_limit_exceeded';

    const upstreamMessage =
      data?.error?.message ||
      data?.message ||
      error.message ||
      '';

    const lowerMsg = upstreamMessage.toLowerCase();

    if (
      lowerMsg.includes('quota') ||
      lowerMsg.includes('billing')
    ) {
      rateLimitType = 'quota_exceeded';
    } else if (
      lowerMsg.includes('concurrent') ||
      lowerMsg.includes('too many requests in progress')
    ) {
      rateLimitType = 'concurrency_limit';
    }

    return res.status(429).json({
      error: {
        message: 'NVIDIA NIM rate limit exceeded',
        type: rateLimitType,
        code: 429,

        details: {
          upstream_message: upstreamMessage,
          retry_after: retryAfter,
          request_id: requestId,

          limits: {
            requests_limit: limitRequests || null,
            requests_remaining: remainingRequests || null,
            tokens_limit: limitTokens || null,
            tokens_remaining: remainingTokens || null
          },

          suggestions: [
            'Reduce request frequency',
            'Lower max_tokens',
            'Retry with exponential backoff',
            'Use a smaller model',
            'Enable request queueing'
          ]
        }
      }
    });
  }

  // Generic error handling
  res.status(status).json({
    error: {
      message:
        data?.error?.message ||
        data?.message ||
        error.message ||
        'Internal server error',

      type:
        data?.error?.type ||
        'invalid_request_error',

      code: status,

      details: {
        upstream: data || null
      }
    }
  });
}
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});
