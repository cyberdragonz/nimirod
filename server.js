// server.js
// OpenAI-compatible proxy for NVIDIA NIM
// Improved DeepSeek streaming stability
// - safer SSE parser
// - timeout support
// - better DeepSeek handling
// - proper stream cleanup
// - safer max_tokens defaults
// - improved 429 handling

const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({
  limit: '50mb'
}));

app.use(express.urlencoded({
  extended: true,
  limit: '50mb'
}));

// NVIDIA NIM config
const NIM_API_BASE =
  process.env.NIM_API_BASE ||
  'https://integrate.api.nvidia.com/v1';

const NIM_API_KEY = process.env.NIM_API_KEY;

// Toggles
const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = false;

// Model mapping
const MODEL_MAPPING = {
  deepseek: 'deepseek-ai/deepseek-v4-pro',
  'deepseek-flash': 'deepseek-ai/deepseek-v4-flash',
  'gpt-oss': 'openai/gpt-oss-120b',
  'glm': 'z-ai/glm-5.2',
  'kimi': 'moonshotai/kimi-k2.6',
  'step': 'stepfun-ai/step-3.5-flash'
};

// --------------------------------------------------
// Helpers
// --------------------------------------------------

function resolveModel(model = '') {
  if (MODEL_MAPPING[model]) {
    return MODEL_MAPPING[model];
  }

  const modelLower = model.toLowerCase();

  if (
    modelLower.includes('gpt-4') ||
    modelLower.includes('claude-opus') ||
    modelLower.includes('405b')
  ) {
    return 'meta/llama-3.1-405b-instruct';
  }

  if (
    modelLower.includes('claude') ||
    modelLower.includes('gemini') ||
    modelLower.includes('70b')
  ) {
    return 'meta/llama-3.1-70b-instruct';
  }

  return 'meta/llama-3.1-8b-instruct';
}

function createNimRequest(body, nimModel) {
  const isDeepSeek = nimModel.includes('deepseek');

  const request = {
    model: nimModel,
    messages: body.messages || [],
    temperature: body.temperature ?? 0.6,

    // safer defaults
    max_tokens: Math.min(body.max_tokens || 1024, 4096),

    stream: !!body.stream
  };

  // thinking mode can freeze some DeepSeek variants
  if (ENABLE_THINKING_MODE) {
    request.extra_body = {
      chat_template_kwargs: {
        thinking: true
      }
    };
  }

  return request;
}

function buildOpenAIResponse(originalModel, data) {
  return {
    id: `chatcmpl-${Date.now()}`,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model: originalModel,

    choices: (data.choices || []).map(choice => {
      let content = choice.message?.content || '';

      if (
        SHOW_REASONING &&
        choice.message?.reasoning_content
      ) {
        content =
          `<think>\n${choice.message.reasoning_content}\n</think>\n\n` +
          content;
      }

      return {
        index: choice.index,
        message: {
          role: choice.message?.role || 'assistant',
          content
        },
        finish_reason: choice.finish_reason
      };
    }),

    usage: data.usage || {
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0
    }
  };
}

// --------------------------------------------------
// Health
// --------------------------------------------------

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// --------------------------------------------------
// Models
// --------------------------------------------------

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

// --------------------------------------------------
// Streaming handler
// --------------------------------------------------

function handleStreamingResponse(
  req,
  res,
  upstreamStream
) {
  res.setHeader(
    'Content-Type',
    'text/event-stream'
  );

  res.setHeader(
    'Cache-Control',
    'no-cache'
  );

  res.setHeader(
    'Connection',
    'keep-alive'
  );

  upstreamStream.setEncoding('utf8');

  let accumulated = '';
  let reasoningStarted = false;

  upstreamStream.on('data', chunk => {
    accumulated += chunk;

    while (true) {
      const separatorIndex =
        accumulated.indexOf('\n\n');

      if (separatorIndex === -1) {
        break;
      }

      const rawEvent =
        accumulated.slice(0, separatorIndex);

      accumulated =
        accumulated.slice(separatorIndex + 2);

      const lines = rawEvent.split('\n');

      for (const line of lines) {
        if (!line.startsWith('data: ')) {
          continue;
        }

        const payload =
          line.slice(6).trim();

        // done
        if (payload === '[DONE]') {
          res.write('data: [DONE]\n\n');
          res.end();
          return;
        }

        try {
          const parsed =
            JSON.parse(payload);

          const delta =
            parsed.choices?.[0]?.delta;

          if (delta) {
            const reasoning =
              delta.reasoning_content;

            const content =
              delta.content;

            if (SHOW_REASONING) {
              let merged = '';

              if (reasoning) {
                if (!reasoningStarted) {
                  merged += '<think>\n';
                  reasoningStarted = true;
                }

                merged += reasoning;
              }

              if (content) {
                if (reasoningStarted) {
                  merged +=
                    '\n</think>\n\n';

                  reasoningStarted = false;
                }

                merged += content;
              }

              delta.content = merged;
            } else {
              delta.content =
                content || '';
            }

            delete delta.reasoning_content;
          }

          res.write(
            `data: ${JSON.stringify(parsed)}\n\n`
          );
        } catch (err) {
          console.error(
            'Stream parse error:',
            err.message
          );
        }
      }
    }
  });

  upstreamStream.on('end', () => {
    res.end();
  });

  upstreamStream.on('error', err => {
    console.error(
      'Upstream stream error:',
      err.message
    );

    if (!res.headersSent) {
      res.status(500).end();
    } else {
      res.end();
    }
  });

  // cleanup if client disconnects
  req.on('close', () => {
    upstreamStream.destroy();
  });
}

// --------------------------------------------------
// Chat completions
// --------------------------------------------------

app.post(
  '/v1/chat/completions',
  async (req, res) => {
    try {
      const {
        model,
        stream = false
      } = req.body;

      const nimModel =
        resolveModel(model);

      const nimRequest =
        createNimRequest(
          req.body,
          nimModel
        );

      const response = await axios.post(
        `${NIM_API_BASE}/chat/completions`,
        nimRequest,
        {
          headers: {
            Authorization: `Bearer ${NIM_API_KEY}`,
            'Content-Type':
              'application/json'
          },

          responseType: stream
            ? 'stream'
            : 'json',

          // prevents hanging forever
          timeout: 1000 * 256
        }
      );

      // streaming
      if (stream) {
        return handleStreamingResponse(
          req,
          res,
          response.data
        );
      }

      // normal response
      const openaiResponse =
        buildOpenAIResponse(
          model,
          response.data
        );

      return res.json(openaiResponse);
    } catch (error) {
      console.error(
        'Proxy error:',
        error.message
      );

      const status =
        error.response?.status || 500;

      const data =
        error.response?.data || {};

      const headers =
        error.response?.headers || {};

      // ------------------------------------------
      // 429 handling
      // ------------------------------------------

      if (status === 429) {
        return res.status(429).json({
          error: {
            message:
              'NVIDIA NIM rate limit exceeded',

            type:
              'rate_limit_exceeded',

            code: 429,

            details: {
              retry_after:
                headers['retry-after'] ||
                null,

              request_id:
                headers['x-request-id'] ||
                null,

              limits: {
                requests_remaining:
                  headers[
                    'x-ratelimit-remaining-requests'
                  ] || null,

                tokens_remaining:
                  headers[
                    'x-ratelimit-remaining-tokens'
                  ] || null
              },

              upstream:
                data?.error?.message ||
                data?.message ||
                null
            }
          }
        });
      }

      // ------------------------------------------
      // generic errors
      // ------------------------------------------

      return res.status(status).json({
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
  }
);

// --------------------------------------------------
// Catch-all
// --------------------------------------------------

app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

// --------------------------------------------------
// Start server
// --------------------------------------------------

app.listen(PORT, () => {
  console.log(
    `NVIDIA NIM proxy running on port ${PORT}`
  );

  console.log(
    `Health: http://localhost:${PORT}/health`
  );

  console.log(
    `Reasoning display: ${
      SHOW_REASONING
        ? 'ENABLED'
        : 'DISABLED'
    }`
  );

  console.log(
    `Thinking mode: ${
      ENABLE_THINKING_MODE
        ? 'ENABLED'
        : 'DISABLED'
    }`
  );
});
