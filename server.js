// server.js
// OpenAI-compatible proxy for NVIDIA NIM
// Fixes: token floor, stream cleanup, SSE flushing, keepalive, thinking toggle

const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));

// NVIDIA NIM config
const NIM_API_BASE =
  process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Toggles
const SHOW_REASONING = true;
const ENABLE_THINKING_MODE = false;

// Token limits (override with env vars if your model has different limits)
// MIN_TOKENS: client values below this are raised to it
// MAX_TOKENS: hard ceiling; going over the model's real limit makes NIM return 400
const MIN_TOKENS = parseInt(process.env.MIN_TOKENS, 10) || 32768;
const MAX_TOKENS = parseInt(process.env.MAX_TOKENS, 10) || 65536;

// Keepalive interval for SSE (prevents host/proxy idle timeouts)
const KEEPALIVE_MS = 15000;

// --------------------------------------------------
// Helpers
// --------------------------------------------------

function resolveModel(model = '') {
  return model;
}

function createNimRequest(body, nimModel) {
  const requested = Number(body.max_tokens) || 0;

  const request = {
    model: nimModel,
    messages: body.messages || [],
    temperature: body.temperature ?? 0.6,
    max_tokens: Math.min(Math.max(requested, MIN_TOKENS), MAX_TOKENS),
    stream: !!body.stream,
    chat_template_kwargs: {
      thinking: ENABLE_THINKING_MODE
    }
  };

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

      if (SHOW_REASONING && choice.message?.reasoning_content) {
        content =
          `<think>\n${choice.message.reasoning_content}\n</think>\n\n` +
          content;
      }

      if (choice.finish_reason) {
        console.log(`finish_reason: ${choice.finish_reason}`);
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

// Reads an axios stream error body into a parsed object/string
async function readStreamBody(stream) {
  return new Promise(resolve => {
    let buf = '';
    stream.setEncoding('utf8');
    stream.on('data', c => (buf += c));
    stream.on('end', () => {
      try {
        resolve(JSON.parse(buf));
      } catch {
        resolve(buf || null);
      }
    });
    stream.on('error', () => resolve(null));
  });
}

// --------------------------------------------------
// Health
// --------------------------------------------------

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    min_tokens: MIN_TOKENS,
    max_tokens: MAX_TOKENS
  });
});

// --------------------------------------------------
// Models
// --------------------------------------------------

app.get('/v1/models', async (req, res) => {
  try {
    const response = await axios.get(`${NIM_API_BASE}/models`, {
      headers: { Authorization: `Bearer ${NIM_API_KEY}` }
    });
    res.json(response.data);
  } catch (error) {
    console.error('Models fetch error:', error.message);
    res.status(500).json({
      error: {
        message: 'Failed to fetch models from NIM',
        type: 'internal_error',
        code: 500
      }
    });
  }
});

// --------------------------------------------------
// Streaming handler
// --------------------------------------------------

function handleStreamingResponse(req, res, upstreamStream) {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no'); // disable nginx buffering
  res.flushHeaders();

  upstreamStream.setEncoding('utf8');

  let accumulated = '';
  let reasoningStarted = false;
  let finished = false;

  // keepalive comment so idle timeouts don't kill long "thinking" gaps
  const keepAlive = setInterval(() => {
    if (!finished && !res.writableEnded) {
      res.write(': keepalive\n\n');
    }
  }, KEEPALIVE_MS);

  function finish() {
    if (finished) return;
    finished = true;
    clearInterval(keepAlive);
    if (!res.writableEnded) res.end();
  }

  // Process one complete SSE event. Returns true if [DONE] was seen.
  function processEvent(rawEvent) {
    const lines = rawEvent.split('\n');

    for (const line of lines) {
      if (!line.startsWith('data:')) continue;

      const payload = line.slice(5).trim();
      if (!payload) continue;

      if (payload === '[DONE]') {
        // close an unfinished think block so clients don't show it open forever
        if (reasoningStarted && SHOW_REASONING) {
          res.write(
            `data: ${JSON.stringify({
              choices: [{ index: 0, delta: { content: '\n</think>\n\n' } }]
            })}\n\n`
          );
          reasoningStarted = false;
        }
        res.write('data: [DONE]\n\n');
        return true;
      }

      try {
        const parsed = JSON.parse(payload);
        const choice = parsed.choices?.[0];
        const delta = choice?.delta;

        if (choice?.finish_reason) {
          console.log(`finish_reason: ${choice.finish_reason}`);
        }

        if (delta) {
          const reasoning = delta.reasoning_content;
          const content = delta.content;

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
                merged += '\n</think>\n\n';
                reasoningStarted = false;
              }
              merged += content;
            }

            delta.content = merged;
          } else {
            delta.content = content || '';
          }

          delete delta.reasoning_content;
        }

        res.write(`data: ${JSON.stringify(parsed)}\n\n`);
      } catch (err) {
        console.error('Stream parse error:', err.message);
      }
    }

    return false;
  }

  upstreamStream.on('data', chunk => {
    if (finished) return;

    // normalize CRLF so the \n\n separator always matches
    accumulated += chunk.replace(/\r\n/g, '\n');

    let separatorIndex;
    while ((separatorIndex = accumulated.indexOf('\n\n')) !== -1) {
      const rawEvent = accumulated.slice(0, separatorIndex);
      accumulated = accumulated.slice(separatorIndex + 2);

      if (processEvent(rawEvent)) {
        finish();
        upstreamStream.destroy();
        return;
      }
    }
  });

  upstreamStream.on('end', () => {
    // flush any final event that had no trailing blank line
    if (!finished && accumulated.trim()) {
      processEvent(accumulated);
      accumulated = '';
    }
    finish();
  });

  upstreamStream.on('error', err => {
    console.error('Upstream stream error:', err.message);
    finish();
  });

  // Client disconnected: only clean up if we didn't finish normally.
  // (Use res, not req: req 'close' can fire as soon as the request body is read.)
  res.on('close', () => {
    clearInterval(keepAlive);
    if (!finished) {
      finished = true;
      upstreamStream.destroy();
    }
  });
}

// --------------------------------------------------
// Chat completions
// --------------------------------------------------

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, stream = false } = req.body;

    const nimModel = resolveModel(model);
    const nimRequest = createNimRequest(req.body, nimModel);

    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        responseType: stream ? 'stream' : 'json',
        timeout: 1000 * 60 * 10
      }
    );

    if (stream) {
      return handleStreamingResponse(req, res, response.data);
    }

    return res.json(buildOpenAIResponse(model, response.data));
  } catch (error) {
    console.error('Proxy error:', error.message);

    const status = error.response?.status || 500;
    const headers = error.response?.headers || {};
    let data = error.response?.data || {};

    // when responseType is 'stream', error bodies arrive as streams
    if (data && typeof data.on === 'function') {
      data = (await readStreamBody(data)) || {};
    }

    // can't send a JSON error if streaming already started
    if (res.headersSent) {
      return res.end();
    }

    // 429 handling
    if (status === 429) {
      return res.status(429).json({
        error: {
          message: 'NVIDIA NIM rate limit exceeded',
          type: 'rate_limit_exceeded',
          code: 429,
          details: {
            retry_after: headers['retry-after'] || null,
            request_id: headers['x-request-id'] || null,
            limits: {
              requests_remaining:
                headers['x-ratelimit-remaining-requests'] || null,
              tokens_remaining:
                headers['x-ratelimit-remaining-tokens'] || null
            },
            upstream: data?.error?.message || data?.message || null
          }
        }
      });
    }

    // generic errors
    return res.status(status).json({
      error: {
        message:
          data?.error?.message ||
          data?.message ||
          error.message ||
          'Internal server error',
        type: data?.error?.type || 'invalid_request_error',
        code: status,
        details: { upstream: data || null }
      }
    });
  }
});

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
  console.log(`NVIDIA NIM proxy running on port ${PORT}`);
  console.log(`Health: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Max tokens: floor ${MIN_TOKENS}, ceiling ${MAX_TOKENS}`);
});
