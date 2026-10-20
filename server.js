// server.js - OpenAI to NVIDIA NIM API Proxy (optimized for reliability)
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const https = require('https');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = false; // Set to true to enable chat_template_kwargs thinking parameter

// 🔧 RELIABILITY SETTINGS
const DEFAULT_MAX_TOKENS = 9024;     // Used only when the app doesn't send its own max_tokens
const MAX_RETRIES = 2;               // How many times to retry the SAME model on a temporary error
const RETRY_BASE_DELAY_MS = 1500;    // Wait before 1st retry (doubles each time)
const REQUEST_TIMEOUT_MS = 900000;   // Give up if NVIDIA goes silent for 5 minutes
const ENABLE_FALLBACK = false;        // If the chosen model keeps failing, try the backup model below
const FALLBACK_MODEL = 'meta/llama-3.1-70b-instruct'; // Backup model (128k context)

// Model mapping (adjust based on available NIM models)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'stepfun-ai/step-5-preview',
  'gpt-4': 'nvidia/nemotron-3.5-super-vl-120b-a12b',
  'gpt-4-turbo': 'nvidia/nemotron-3-ultra-550b-a55b',
  'gpt-4o': 'deepseek-ai/deepseek-v4.1-flash',
  'claude-3-opus': 'z-ai/glm-5.3-flash',
  'claude-3-sonnet': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking'
};

// Reusable connection to NVIDIA (faster than opening a new one every time)
const nim = axios.create({
  baseURL: NIM_API_BASE,
  timeout: REQUEST_TIMEOUT_MS,
  httpsAgent: new https.Agent({ keepAlive: true }),
  maxBodyLength: Infinity,
  maxContentLength: Infinity,
  headers: {
    'Authorization': `Bearer ${NIM_API_KEY}`,
    'Content-Type': 'application/json'
  }
});

// ---------- Helpers ----------

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const NETWORK_ERRORS = ['ECONNRESET', 'ETIMEDOUT', 'EAI_AGAIN', 'ENOTFOUND', 'EPIPE'];
const RETRYABLE_STATUS = [408, 429, 500, 502, 503, 504];

// Temporary problems worth retrying on the same model
function shouldRetry(err) {
  const status = err.response?.status;
  if (status) return RETRYABLE_STATUS.includes(status);
  return NETWORK_ERRORS.includes(err.code);
}

// Problems where switching to the backup model might help
function shouldFallback(err) {
  const status = err.response?.status;
  return shouldRetry(err) || status === 404 || status === 410;
}

// Pull the real error message out of NVIDIA's reply (so you see WHY it failed)
async function extractErrorMessage(error) {
  const data = error.response?.data;
  try {
    if (data && typeof data.on === 'function') {
      const chunks = [];
      for await (const c of data) chunks.push(c);
      const text = Buffer.concat(chunks).toString('utf8');
      try {
        const j = JSON.parse(text);
        return j.error?.message || j.detail || j.message || text;
      } catch (e) {
        return text || error.message;
      }
    }
    if (data) return data.error?.message || data.detail || data.message || JSON.stringify(data);
  } catch (e) {}
  return error.message;
}

// Send the request, automatically retrying temporary failures
async function postWithRetry(body, stream, signal) {
  let attempt = 0;
  while (true) {
    try {
      return await nim.post('/chat/completions', body, {
        responseType: stream ? 'stream' : 'json',
        signal
      });
    } catch (err) {
      if (signal?.aborted) throw err;
      if (!shouldRetry(err) || attempt >= MAX_RETRIES) throw err;

      if (err.response?.data?.destroy) err.response.data.destroy();

      const retryAfter = parseInt(err.response?.headers?.['retry-after'], 10);
      const delay = Number.isFinite(retryAfter)
        ? Math.min(retryAfter * 1000, 10000)
        : RETRY_BASE_DELAY_MS * Math.pow(2, attempt) + Math.floor(Math.random() * 500);

      console.warn(`[retry ${attempt + 1}/${MAX_RETRIES}] ${body.model} -> ${err.response?.status || err.code}, waiting ${delay}ms`);
      await sleep(delay);
      attempt++;
    }
  }
}

// Pick the NVIDIA model for whatever name the app sends (no extra test request needed)
function resolveModel(requested) {
  if (MODEL_MAPPING[requested]) return MODEL_MAPPING[requested];
  if (typeof requested === 'string' && requested.includes('/')) return requested; // already a NIM model name

  const m = String(requested || '').toLowerCase();
  if (m.includes('gpt-4') || m.includes('claude-opus') || m.includes('405b')) {
    return 'meta/llama-3.1-405b-instruct';
  } else if (m.includes('claude') || m.includes('gemini') || m.includes('70b')) {
    return 'meta/llama-3.1-70b-instruct';
  }
  return 'meta/llama-3.1-8b-instruct';
}

// ---------- Routes ----------

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    fallback_enabled: ENABLE_FALLBACK,
    uptime_seconds: Math.round(process.uptime())
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map((model) => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));

  res.json({ object: 'list', data: models });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  // If the app disconnects/cancels, stop the request to NVIDIA too
  const controller = new AbortController();
  res.on('close', () => {
    if (!res.writableFinished) controller.abort();
  });

  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    const isStream = !!stream;
    const nimModel = resolveModel(model);

    const baseRequest = {
      messages: messages,
      temperature: temperature ?? 0.6,
      max_tokens: max_tokens || DEFAULT_MAX_TOKENS,
      stream: isStream,
      ...(ENABLE_THINKING_MODE ? { chat_template_kwargs: { thinking: true } } : {})
    };

    // Primary model first, then optional backup
    const candidates = [nimModel];
    if (ENABLE_FALLBACK && FALLBACK_MODEL && FALLBACK_MODEL !== nimModel) {
      candidates.push(FALLBACK_MODEL);
    }

    let response;
    let lastError;
    for (let i = 0; i < candidates.length; i++) {
      try {
        response = await postWithRetry({ ...baseRequest, model: candidates[i] }, isStream, controller.signal);
        if (i > 0) console.warn(`Fallback used: ${nimModel} -> ${candidates[i]}`);
        break;
      } catch (err) {
        if (controller.signal.aborted) return;
        err.nimMessage = await extractErrorMessage(err);
        console.error(`[${candidates[i]}] failed: ${err.response?.status || err.code || 'unknown'} - ${err.nimMessage}`);
        lastError = err;
        if (!shouldFallback(err)) break;
      }
    }

    if (!response) throw lastError;

    if (isStream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache, no-transform');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no');
      res.flushHeaders();

      let buffer = '';
      let reasoningStarted = false;

      const processLine = (rawLine) => {
        const line = rawLine.replace(/\r$/, '');
        if (!line.startsWith('data: ')) return;

        if (line.includes('[DONE]')) {
          res.write('data: [DONE]\n\n');
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
              data.choices[0].delta.content = content ? content : '';
              delete data.choices[0].delta.reasoning_content;
            }
          }
          res.write(`data: ${JSON.stringify(data)}\n\n`);
        } catch (e) {
          res.write(line + '\n\n');
        }
      };

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        lines.forEach(processLine);
      });

      response.data.on('end', () => {
        if (buffer) processLine(buffer);
        res.end();
      });

      response.data.on('error', (err) => {
        console.error('Stream error:', err.message);
        res.end();
      });
    } else {
      // Transform NIM response to OpenAI format with reasoning
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map((choice) => {
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
    if (controller.signal.aborted) return;

    const status = error.response?.status || (error.code === 'ECONNABORTED' ? 504 : 500);
    const message = error.nimMessage || error.message || 'Internal server error';
    console.error(`Proxy error (${status}):`, message);

    if (res.headersSent) return res.end();

    res.status(status).json({
      error: {
        message: message,
        type: 'invalid_request_error',
        code: status
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

const server = app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Fallback model: ${ENABLE_FALLBACK ? FALLBACK_MODEL : 'DISABLED'}`);
});

// Keep connections open a bit longer than hosting proxies do (prevents random 502s)
server.keepAliveTimeout = 120000;
server.headersTimeout = 125000;
