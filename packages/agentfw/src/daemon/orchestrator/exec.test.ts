import { describe, expect, it } from 'vitest'
import type { ProviderEntry } from '../../core/model-registry.ts'
import {
  clampOutputBudget,
  clampOutputBudgetBytes,
  clarifyUpstreamError,
  dropSwapIncompatibleBetas,
  generationUrl,
  parseOverflowNumbers,
  safeRetryBudget,
  isAnthropicNative,
  isGenerationPath,
  rewriteModel,
  stripAnthropicServerToolsFromBody,
  withoutServerTools,
} from './exec.ts'

function encode(value: unknown): ArrayBuffer {
  const u8 = new TextEncoder().encode(typeof value === 'string' ? value : JSON.stringify(value))
  const ab = new ArrayBuffer(u8.byteLength)
  new Uint8Array(ab).set(u8)
  return ab
}

function decode(buf: ArrayBuffer): unknown {
  return JSON.parse(new TextDecoder().decode(buf))
}

describe('dropSwapIncompatibleBetas', () => {
  const beta = (h: Headers) => h.get('anthropic-beta')

  it('removes the 1M long-context beta, keeps the rest', () => {
    const h = new Headers({ 'anthropic-beta': 'prompt-caching-2024-07-31,context-1m-2025-08-07' })
    dropSwapIncompatibleBetas(h)
    expect(beta(h)).toBe('prompt-caching-2024-07-31')
  })

  it('deletes the header when only context-1m was present', () => {
    const h = new Headers({ 'anthropic-beta': 'context-1m-2025-08-07' })
    dropSwapIncompatibleBetas(h)
    expect(h.has('anthropic-beta')).toBe(false)
  })

  it('is a no-op when there is no anthropic-beta header', () => {
    const h = new Headers({ 'x-api-key': 'k' })
    dropSwapIncompatibleBetas(h)
    expect(h.has('anthropic-beta')).toBe(false)
  })
})

describe('generationUrl', () => {
  it('appends the version segment to a host-root base', () => {
    expect(generationUrl('https://api.anthropic.com', 'anthropic-messages')).toBe(
      'https://api.anthropic.com/v1/messages',
    )
    expect(generationUrl('https://api.openai.com', 'openai-chat')).toBe(
      'https://api.openai.com/v1/chat/completions',
    )
    expect(generationUrl('https://api.openai.com', 'openai-responses')).toBe(
      'https://api.openai.com/v1/responses',
    )
  })

  it('does not double the version segment when the base already ends in /v1', () => {
    expect(generationUrl('https://api.xiangxinai.cn/coding/v1', 'openai-chat')).toBe(
      'https://api.xiangxinai.cn/coding/v1/chat/completions',
    )
    expect(generationUrl('https://openrouter.ai/api/v1', 'openai-chat')).toBe(
      'https://openrouter.ai/api/v1/chat/completions',
    )
  })

  it('tolerates a trailing slash', () => {
    expect(generationUrl('https://api.anthropic.com/', 'anthropic-messages')).toBe(
      'https://api.anthropic.com/v1/messages',
    )
    expect(generationUrl('https://api.openai.com/v1/', 'openai-chat')).toBe(
      'https://api.openai.com/v1/chat/completions',
    )
  })

  it('omits the /v1 segment on codex chatgpt.com backend', () => {
    // chatgpt.com/backend-api/codex serves /responses directly; the codex
    // CLI hits this path natively. Without this exception, off-agent
    // routes (openclaw → codex subscription) build /v1/responses and the
    // upstream replies with a 403 HTML login page.
    expect(generationUrl('https://chatgpt.com/backend-api/codex', 'openai-responses')).toBe(
      'https://chatgpt.com/backend-api/codex/responses',
    )
    expect(generationUrl('https://chatgpt.com/backend-api/codex/', 'openai-responses')).toBe(
      'https://chatgpt.com/backend-api/codex/responses',
    )
  })
})

describe('isGenerationPath', () => {
  it('matches each API generation endpoint', () => {
    expect(isGenerationPath('anthropic-messages', '/v1/messages')).toBe(true)
    expect(isGenerationPath('openai-chat', '/v1/chat/completions')).toBe(true)
    expect(isGenerationPath('openai-chat', '/chat/completions')).toBe(true)
    expect(isGenerationPath('openai-responses', '/v1/responses')).toBe(true)
  })

  it('rejects non-generation and mismatched paths', () => {
    expect(isGenerationPath('anthropic-messages', '/v1/models')).toBe(false)
    expect(isGenerationPath('openai-chat', '/v1/messages')).toBe(false)
    expect(isGenerationPath('openai-responses', '/v1/chat/completions')).toBe(false)
  })
})

describe('rewriteModel', () => {
  it('replaces the top-level model field', () => {
    const out = rewriteModel(encode({ model: 'old', messages: [] }), 'new-model')
    expect(decode(out)).toEqual({ model: 'new-model', messages: [] })
  })

  it('adds a model field when absent', () => {
    const out = rewriteModel(encode({ messages: [] }), 'new-model')
    expect(decode(out)).toEqual({ messages: [], model: 'new-model' })
  })

  it('forwards an unparseable body unchanged', () => {
    const buf = encode('not json{')
    expect(rewriteModel(buf, 'new-model')).toBe(buf)
  })

  it('forwards a non-object body unchanged', () => {
    const buf = encode([1, 2, 3])
    expect(rewriteModel(buf, 'new-model')).toBe(buf)
  })
})

function provider(baseUrl: string): ProviderEntry {
  return {
    id: 'p',
    label: 'p',
    baseUrl,
    api: 'anthropic-messages',
    auth: { kind: 'passthrough' },
    origin: 'manual',
  }
}

describe('isAnthropicNative', () => {
  it('recognizes api.anthropic.com and *.anthropic.com', () => {
    expect(isAnthropicNative(provider('https://api.anthropic.com'))).toBe(true)
    expect(isAnthropicNative(provider('https://api.anthropic.com/v1'))).toBe(true)
    expect(isAnthropicNative(provider('https://shadow.anthropic.com'))).toBe(true)
  })

  it('rejects third-party Anthropic-compatible endpoints', () => {
    expect(isAnthropicNative(provider('https://api.deepseek.com/anthropic'))).toBe(false)
    expect(isAnthropicNative(provider('https://api.example.com/'))).toBe(false)
  })

  it('returns false for a malformed baseUrl', () => {
    expect(isAnthropicNative(provider('not a url'))).toBe(false)
  })
})

describe('withoutServerTools', () => {
  const serverTool = { type: 'web_search_20250305', name: 'web_search' }
  const customTool = { name: 'my_tool', input_schema: { type: 'object' } }
  const mcpTool = { name: 'mcp__agentfw__web_search', input_schema: { type: 'object' } }

  it('drops Anthropic server tools, keeps custom + mcp tools', () => {
    const body = { messages: [], tools: [serverTool, customTool, mcpTool] }
    const out = withoutServerTools(body)
    expect(out).not.toBe(body)
    expect(out.tools).toEqual([customTool, mcpTool])
  })

  it('returns the same reference when there is nothing to strip', () => {
    const body = { messages: [], tools: [customTool] }
    expect(withoutServerTools(body)).toBe(body)
  })

  it('removes the tools key entirely when every entry was a server tool', () => {
    const body = { messages: [], tools: [serverTool] }
    const out = withoutServerTools(body)
    expect('tools' in out).toBe(false)
  })

  it('treats type:"custom" as a custom tool (not stripped)', () => {
    const c = { type: 'custom', name: 'x', input_schema: { type: 'object' } }
    const body = { tools: [c] }
    expect(withoutServerTools(body)).toBe(body)
  })

  it('leaves bodies without tools unchanged', () => {
    const body = { messages: [] }
    expect(withoutServerTools(body)).toBe(body)
  })
})

describe('clampOutputBudget', () => {
  it('is a no-op when neither contextWindow nor maxTokens is set', () => {
    const body: Record<string, unknown> = { max_tokens: 32000, messages: [] }
    expect(clampOutputBudget(body, 'openai-chat', { id: 'm' })).toBe(false)
    expect(body.max_tokens).toBe(32000)
  })

  it('caps an oversized budget to the model maxTokens', () => {
    const body: Record<string, unknown> = { max_tokens: 32000, messages: [] }
    expect(clampOutputBudget(body, 'anthropic-messages', { id: 'm', maxTokens: 8192 })).toBe(true)
    expect(body.max_tokens).toBe(8192)
  })

  it('leaves a budget already within the limit untouched', () => {
    const body: Record<string, unknown> = { max_tokens: 4096, messages: [] }
    expect(clampOutputBudget(body, 'anthropic-messages', { id: 'm', maxTokens: 8192 })).toBe(false)
    expect(body.max_tokens).toBe(4096)
  })

  it('shrinks output so input + output fits the context window', () => {
    // ~30 k chars of input ≈ 8.5 k estimated tokens; window 16 k leaves
    // well under the requested 32 k of output.
    const big = 'x'.repeat(30_000)
    const body: Record<string, unknown> = { max_tokens: 32000, messages: [{ content: big }] }
    expect(clampOutputBudget(body, 'openai-chat', { id: 'm', contextWindow: 16000 })).toBe(true)
    const out = body.max_tokens as number
    const estIn = Math.ceil(JSON.stringify(body).length / 3.5)
    expect(estIn + out).toBeLessThanOrEqual(16000)
    expect(out).toBeGreaterThanOrEqual(1024)
  })

  it('floors at MIN_OUTPUT_TOKENS even when input alone nearly fills the window', () => {
    const huge = 'x'.repeat(80_000)
    const body: Record<string, unknown> = { max_tokens: 32000, messages: [{ content: huge }] }
    expect(clampOutputBudget(body, 'openai-chat', { id: 'm', contextWindow: 16000 })).toBe(true)
    expect(body.max_tokens).toBe(1024)
  })

  it('never inflates a budget the client did not request', () => {
    const body: Record<string, unknown> = { messages: [] }
    expect(clampOutputBudget(body, 'openai-chat', { id: 'm', contextWindow: 131072 })).toBe(false)
    expect('max_tokens' in body).toBe(false)
  })

  it('clamps max_completion_tokens for openai-chat too', () => {
    const body: Record<string, unknown> = { max_completion_tokens: 32000, messages: [] }
    expect(clampOutputBudget(body, 'openai-chat', { id: 'm', maxTokens: 8192 })).toBe(true)
    expect(body.max_completion_tokens).toBe(8192)
  })

  it('clamps max_output_tokens for openai-responses', () => {
    const body: Record<string, unknown> = { max_output_tokens: 32000, messages: [] }
    expect(clampOutputBudget(body, 'openai-responses', { id: 'm', maxTokens: 8192 })).toBe(true)
    expect(body.max_output_tokens).toBe(8192)
  })
})

describe('clampOutputBudgetBytes', () => {
  it('round-trips through bytes and clamps the budget', () => {
    const out = clampOutputBudgetBytes(
      encode({ model: 'm', max_tokens: 32000, messages: [] }),
      'anthropic-messages',
      { id: 'm', maxTokens: 8192 },
    )
    expect((decode(out) as { max_tokens: number }).max_tokens).toBe(8192)
  })

  it('returns the same buffer when nothing to clamp', () => {
    const body = encode({ max_tokens: 1000, messages: [] })
    expect(clampOutputBudgetBytes(body, 'anthropic-messages', { id: 'm', maxTokens: 8192 })).toBe(
      body,
    )
  })

  it('returns the same buffer when no limits configured', () => {
    const body = encode({ max_tokens: 32000, messages: [] })
    expect(clampOutputBudgetBytes(body, 'anthropic-messages', { id: 'm' })).toBe(body)
  })

  it('returns the same buffer for an unparseable body', () => {
    const body = encode('not json{')
    expect(clampOutputBudgetBytes(body, 'anthropic-messages', { id: 'm', maxTokens: 8 })).toBe(body)
  })
})

describe('clarifyUpstreamError', () => {
  it('rewrites a vLLM context-overflow 400 into an actionable message', () => {
    const vllm =
      "This model's maximum context length is 131072 tokens. However, you " +
      'requested 32000 output tokens and your prompt contains at least 99073 ' +
      'input tokens, for a total of at least 131073 tokens. Please reduce the ' +
      'length of the input prompt or the number of requested output tokens.'
    const out = clarifyUpstreamError(400, vllm)
    expect(out).toContain('context window exceeded')
    expect(out).toContain('/compact')
    expect(out).toContain('131072')
  })

  it('recognizes context_length_exceeded from OpenAI-compatible servers', () => {
    const out = clarifyUpstreamError(400, '{"error":{"code":"context_length_exceeded"}}')
    expect(out).toContain('context window exceeded')
  })

  it('passes a non-overflow error through as a trimmed excerpt', () => {
    const out = clarifyUpstreamError(401, '{"error":"invalid api key"}')
    expect(out).toContain('invalid api key')
    expect(out).not.toContain('context window exceeded')
  })

  it('does not treat overflow wording as overflow on non-4xx statuses', () => {
    const out = clarifyUpstreamError(500, 'maximum context length is 131072')
    expect(out).not.toContain('context window exceeded')
  })

  it('falls back to an HTTP code when the body is empty', () => {
    expect(clarifyUpstreamError(503, '')).toBe('HTTP 503')
  })
})

const VLLM_OVERFLOW =
  "This model's maximum context length is 131072 tokens. However, you " +
  'requested 32000 output tokens and your prompt contains at least 99073 ' +
  'input tokens, for a total of at least 131073 tokens. Please reduce the ' +
  'length of the input prompt or the number of requested output tokens.'

describe('parseOverflowNumbers', () => {
  it('extracts window and input from a vLLM overflow error', () => {
    expect(parseOverflowNumbers(VLLM_OVERFLOW)).toEqual({ window: 131072, input: 99073 })
  })

  it('extracts input from an OpenAI-style "resulted in N tokens" message', () => {
    const msg =
      "This model's maximum context length is 8192 tokens. However, your " +
      'messages resulted in 9000 tokens.'
    expect(parseOverflowNumbers(msg)).toEqual({ window: 8192, input: 9000 })
  })

  it('returns empty when no numbers are present', () => {
    expect(parseOverflowNumbers('invalid api key')).toEqual({})
  })
})

describe('safeRetryBudget', () => {
  it('computes a budget that fits the window from the upstream numbers', () => {
    const budget = safeRetryBudget(VLLM_OVERFLOW, {})
    // 131072 - 99073 - 256 margin
    expect(budget).toBe(131072 - 99073 - 256)
    expect(99073 + (budget ?? 0)).toBeLessThan(131072)
  })

  it('prefers the configured contextWindow over the parsed one', () => {
    const budget = safeRetryBudget(VLLM_OVERFLOW, { contextWindow: 120000 })
    expect(budget).toBe(120000 - 99073 - 256)
  })

  it('returns undefined when input alone leaves no usable room', () => {
    const msg =
      "This model's maximum context length is 131072 tokens. However, your " +
      'prompt contains at least 131000 input tokens.'
    expect(safeRetryBudget(msg, {})).toBeUndefined()
  })

  it('returns undefined when the error carries no numbers', () => {
    expect(safeRetryBudget('rate limited', { contextWindow: 131072 })).toBeUndefined()
  })
})

describe('stripAnthropicServerToolsFromBody', () => {
  it('round-trips through bytes and strips server tools', () => {
    const body = encode({
      model: 'claude-x',
      tools: [
        { type: 'web_search_20250305', name: 'web_search' },
        { name: 'my_tool', input_schema: {} },
      ],
    })
    const out = stripAnthropicServerToolsFromBody(body)
    const obj = decode(out) as { tools: Array<{ name: string }> }
    expect(obj.tools).toHaveLength(1)
    expect(obj.tools[0]?.name).toBe('my_tool')
  })

  it('returns the same buffer when nothing to strip', () => {
    const body = encode({ tools: [{ name: 'x', input_schema: {} }] })
    expect(stripAnthropicServerToolsFromBody(body)).toBe(body)
  })

  it('returns the same buffer for an unparseable body', () => {
    const body = encode('not json{')
    expect(stripAnthropicServerToolsFromBody(body)).toBe(body)
  })
})
