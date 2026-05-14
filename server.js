// server.js
// OpenAI-compatible NVIDIA NIM proxy
// Rewritten + hardened version

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const axiosRetry = require('axios-retry').default;
const PQueue = require('p-queue').default;

const app = express();

const PORT = Number(process.env.PORT) || 3000;

// ======================================================
// MIDDLEWARE
// ======================================================

app.use(cors());

app.use(
  express.json({
    limit: '10mb'
  })
);

// ======================================================
// CONFIG
// ======================================================

const CONFIG = {
  NIM_API_BASE:
    process.env.NIM_API_BASE ||
    'https://integrate.api.nvidia.com/v1',

  NIM_API_KEY:
    process.env.NIM_API_KEY || '',

  SHOW_REASONING: false,

  ENABLE_THINKING_MODE: false,

  REQUEST_TIMEOUT_MS:
    1000 * 60 * 10,

  MAX_CONCURRENT_REQUESTS: 5
};

if (!CONFIG.NIM_API_KEY) {
  console.warn(
    'WARNING: NIM_API_KEY is not set'
  );
}

// ======================================================
// MODEL MAP
// ======================================================

const MODEL_MAPPING = {
  deepseek:
    'deepseek-ai/deepseek-v4-pro',

  'deepseek-flash':
    'deepseek-ai/deepseek-v4-flash'
};

function resolveModel(model = '') {
  if (MODEL_MAPPING[model]) {
    return MODEL_MAPPING[model];
  }

  const lower = String(model)
    .toLowerCase()
    .trim();

  if (
    lower.includes('gpt-4') ||
    lower.includes('claude-opus') ||
    lower.includes('405b')
  ) {
    return 'meta/llama-3.1-405b-instruct';
  }

  if (
    lower.includes('claude') ||
    lower.includes('gemini') ||
    lower.includes('70b')
  ) {
    return 'meta/llama-3.1-70b-instruct';
  }

  return 'meta/llama-3.1-8b-instruct';
}

// ======================================================
// AXIOS CLIENT
// ======================================================

const nim = axios.create({
  baseURL: CONFIG.NIM_API_BASE,

  timeout: CONFIG.REQUEST_TIMEOUT_MS,

  headers: {
    Authorization:
      `Bearer ${CONFIG.NIM_API_KEY}`,

    'Content-Type':
      'application/json'
  }
});

// ======================================================
// RETRY LOGIC
// ======================================================

axiosRetry(nim, {
  retries: 4,

  retryDelay: retryCount => {
    return Math.min(
      1000 * 2 ** retryCount,
      10000
    );
  },

  retryCondition: error => {
    const status =
      error.response?.status;

    return (
      status === 429 ||
      status === 502 ||
      status === 503 ||
      status === 504
    );
  }
});

// ======================================================
// REQUEST QUEUE
// ======================================================

const queue = new PQueue({
  concurrency:
    CONFIG.MAX_CONCURRENT_REQUESTS
});

// ======================================================
// HELPERS
// ======================================================

function buildNimRequest(
  body,
  resolvedModel
) {
  return {
    model: resolvedModel,

    messages: Array.isArray(
      body.messages
    )
      ? body.messages
      : [],

    temperature:
      typeof body.temperature ===
      'number'
        ? body.temperature
        : 0.7,

    top_p:
      typeof body.top_p ===
      'number'
        ? body.top_p
        : 0.95,

    presence_penalty:
      typeof body
        .presence_penalty ===
      'number'
        ? body
            .presence_penalty
        : 0,

    frequency_penalty:
      typeof body
        .frequency_penalty ===
      'number'
        ? body
            .frequency_penalty
        : 0,

    max_tokens: Math.min(
      typeof body.max_tokens ===
        'number'
        ? body.max_tokens
        : 4096,
      8192
    ),

    stream: Boolean(body.stream),

    ...(CONFIG
      .ENABLE_THINKING_MODE && {
      extra_body: {
        chat_template_kwargs: {
          thinking: true
        }
      }
    })
  };
}

function injectReasoning({
  reasoning,
  content,
  reasoningStarted
}) {
  if (!CONFIG.SHOW_REASONING) {
    return {
      text: content || '',
      reasoningStarted
    };
  }

  let text = '';

  if (reasoning) {
    if (!reasoningStarted) {
      text += '<think>\n';
      reasoningStarted = true;
    }

    text += reasoning;
  }

  if (content) {
    if (reasoningStarted) {
      text += '\n</think>\n\n';
      reasoningStarted = false;
    }

    text += content;
  }

  return {
    text,
    reasoningStarted
  };
}

function createOpenAIResponse({
  upstream,
  requestedModel
}) {
  return {
    id:
      `chatcmpl-${Date.now()}`,

    object: 'chat.completion',

    created: Math.floor(
      Date.now() / 1000
    ),

    model: requestedModel,

    choices: (
      upstream.choices || []
    ).map(choice => {
      const reasoning =
        choice.message
          ?.reasoning_content ||
        '';

      const content =
        choice.message?.content ||
        '';

      const finalContent =
        CONFIG.SHOW_REASONING &&
        reasoning
          ? `<think>\n${reasoning}\n</think>\n\n${content}`
          : content;

      return {
        index: choice.index,

        message: {
          role:
            choice.message?.role ||
            'assistant',

          content: finalContent
        },

        finish_reason:
          choice.finish_reason ||
          'stop'
      };
    }),

    usage:
      upstream.usage || {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0
      }
  };
}

function parseRateLimit(error) {
  const status =
    error.response?.status;

  if (status !== 429) {
    return null;
  }

  const data =
    error.response?.data || {};

  const headers =
    error.response?.headers ||
    {};

  const upstreamMessage =
    data?.error?.message ||
    data?.message ||
    error.message ||
    'Rate limit exceeded';

  let type =
    'rate_limit_exceeded';

  const lower =
    upstreamMessage.toLowerCase();

  if (
    lower.includes('quota')
  ) {
    type = 'quota_exceeded';
  }

  if (
    lower.includes(
      'concurrent'
    ) ||
    lower.includes(
      'requests in progress'
    )
  ) {
    type =
      'concurrency_limit';
  }

  return {
    error: {
      message:
        'NVIDIA NIM rate limit exceeded',

      type,

      code: 429,

      details: {
        upstream_message:
          upstreamMessage,

        retry_after:
          headers[
            'retry-after'
          ] || null,

        request_id:
          headers[
            'x-request-id'
          ] || null
      }
    }
  };
}

function createGenericError(
  error
) {
  const status =
    error.response?.status ||
    500;

  const data =
    error.response?.data || {};

  return {
    status,

    body: {
      error: {
        message:
          data?.error
            ?.message ||
          data?.message ||
          error.message ||
          'Internal server error',

        type:
          data?.error?.type ||
          'server_error',

        code: status
      }
    }
  };
}

// ======================================================
// STREAM HANDLER
// ======================================================

async function handleStream({
  response,
  res,
  requestedModel
}) {
  res.setHeader(
    'Content-Type',
    'text/event-stream'
  );

  res.setHeader(
    'Cache-Control',
    'no-cache, no-transform'
  );

  res.setHeader(
    'Connection',
    'keep-alive'
  );

  res.flushHeaders?.();

  let buffer = '';
  let closed = false;
  let reasoningStarted =
    false;

  const heartbeat =
    setInterval(() => {
      if (!closed) {
        res.write(': ping\n\n');
      }
    }, 15000);

  function cleanup() {
    closed = true;

    clearInterval(heartbeat);

    response.data.removeAllListeners();

    if (
      response.data.destroy
    ) {
      response.data.destroy();
    }

    if (
      !res.writableEnded
    ) {
      res.end();
    }
  }

  response.data.on(
    'data',
    chunk => {
      buffer += chunk.toString(
        'utf8'
      );

      while (
        buffer.includes(
          '\n\n'
        )
      ) {
        const idx =
          buffer.indexOf(
            '\n\n'
          );

        const rawEvent =
          buffer.slice(
            0,
            idx
          );

        buffer =
          buffer.slice(
            idx + 2
          );

        const lines =
          rawEvent.split(
            '\n'
          );

        for (const line of lines) {
          if (
            !line.startsWith(
              'data:'
            )
          ) {
            continue;
          }

          const payload =
            line
              .slice(5)
              .trim();

          if (!payload) {
            continue;
          }

          if (
            payload ===
            '[DONE]'
          ) {
            if (
              !res.writableEnded
            ) {
              res.write(
                'data: [DONE]\n\n'
              );
            }

            cleanup();

            return;
          }

          let parsed;

          try {
            parsed =
              JSON.parse(
                payload
              );
          } catch {
            continue;
          }

          parsed.model =
            requestedModel;

          const delta =
            parsed.choices?.[0]
              ?.delta;

          if (delta) {
            const reasoning =
              delta.reasoning_content ||
              '';

            const content =
              delta.content ||
              '';

            const result =
              injectReasoning(
                {
                  reasoning,
                  content,
                  reasoningStarted
                }
              );

            reasoningStarted =
              result.reasoningStarted;

            delta.content =
              result.text;

            delete delta.reasoning_content;
          }

          if (
            !closed &&
            !res.writableEnded
          ) {
            res.write(
              `data: ${JSON.stringify(
                parsed
              )}\n\n`
            );
          }
        }
      }
    }
  );

  response.data.on(
    'end',
    cleanup
  );

  response.data.on(
    'error',
    err => {
      console.error(
        'Streaming error:',
        err.message
      );

      if (
        !res.writableEnded
      ) {
        res.write(
          `data: ${JSON.stringify(
            {
              error: {
                message:
                  'Upstream streaming error',

                type:
                  'stream_error'
              }
            }
          )}\n\n`
        );
      }

      cleanup();
    }
  );

  res.on('close', cleanup);
}

// ======================================================
// ROUTES
// ======================================================

app.get(
  '/health',
  async (req, res) => {
    try {
      await nim.get('/models');

      res.json({
        status: 'ok',
        upstream:
          'reachable',

        reasoning_display:
          CONFIG.SHOW_REASONING,

        thinking_mode:
          CONFIG
            .ENABLE_THINKING_MODE
      });
    } catch {
      res.status(500).json({
        status: 'error',
        upstream:
          'unreachable'
      });
    }
  }
);

app.get(
  '/v1/models',
  async (req, res) => {
    const models =
      Object.keys(
        MODEL_MAPPING
      ).map(id => ({
        id,

        object: 'model',

        created:
          Date.now(),

        owned_by:
          'nvidia-nim-proxy'
      }));

    res.json({
      object: 'list',
      data: models
    });
  }
);

app.post(
  '/v1/chat/completions',
  async (req, res) => {
    const controller =
      new AbortController();

    req.on('close', () => {
      controller.abort();
    });

    try {
      const {
        model = '',
        stream = false
      } = req.body;

      const resolvedModel =
        resolveModel(model);

      const nimRequest =
        buildNimRequest(
          req.body,
          resolvedModel
        );

      const upstream =
        await queue.add(() =>
          nim.post(
            '/chat/completions',
            nimRequest,
            {
              responseType:
                stream
                  ? 'stream'
                  : 'json',

              signal:
                controller.signal
            }
          )
        );

      if (stream) {
        return handleStream({
          response: upstream,
          res,
          requestedModel:
            model
        });
      }

      const transformed =
        createOpenAIResponse(
          {
            upstream:
              upstream.data,

            requestedModel:
              model
          }
        );

      return res.json(
        transformed
      );
    } catch (error) {
      if (
        axios.isCancel(
          error
        )
      ) {
        return;
      }

      console.error(
        'Proxy error:',
        error.response
          ?.data ||
          error.message
      );

      const rateLimit =
        parseRateLimit(
          error
        );

      if (rateLimit) {
        return res
          .status(429)
          .json(
            rateLimit
          );
      }

      const generic =
        createGenericError(
          error
        );

      return res
        .status(
          generic.status
        )
        .json(
          generic.body
        );
    }
  }
);

// ======================================================
// FALLBACK
// ======================================================

app.use((req, res) => {
  res.status(404).json({
    error: {
      message:
        `Endpoint ${req.path} not found`,

      type:
        'invalid_request_error',

      code: 404
    }
  });
});

// ======================================================
// START
// ======================================================

app.listen(PORT, () => {
  console.log('');

  console.log(
    `NVIDIA NIM proxy running on port ${PORT}`
  );

  console.log(
    `Health: http://localhost:${PORT}/health`
  );

  console.log(
    `Reasoning display: ${
      CONFIG.SHOW_REASONING
        ? 'ENABLED'
        : 'DISABLED'
    }`
  );

  console.log(
    `Thinking mode: ${
      CONFIG.ENABLE_THINKING_MODE
        ? 'ENABLED'
        : 'DISABLED'
    }`
  );

  console.log('');
});
