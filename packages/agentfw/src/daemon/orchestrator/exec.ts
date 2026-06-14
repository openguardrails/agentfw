// Upstream-request construction and execution for routed model swaps. The
// generation URL, the model-field rewrite, and provider auth back the Stage 3
// same-protocol tee path; `execAttempt` is the buffered unit the chain
// orchestrator and cross-protocol swaps are built from — one upstream call
// with request translation and response→IR parsing.

import type { AgentId } from '../../core/agent.ts'
import { logger } from '../../core/logger.ts'
import type { ModelApi, ProviderAuth, ProviderEntry } from '../../core/model-registry.ts'
import type { DecoderKind } from '../../core/routes.ts'
import { getSecret } from '../../core/secrets.ts'
import { makeRestoreTransform, maskText, restoreText } from '../proxy/credential-mask.ts'
import { dynamicHeadersFor } from '../proxy/dynamic-headers.ts'
import { filterRequestHeaders } from '../proxy/forward.ts'
import { getSecrets } from '../routing/load.ts'
import {
  adaptForClaudeCodeOAuth,
  adaptForCodexBackend,
  isCodexChatGptBackend,
} from '../translate/codex-compat.ts'
import { type IRResponse, parseResponseToIR, translateRequest } from '../translate/index.ts'
import { getAgentToken } from './oauth/index.ts'
import type { ResolvedMember } from './resolve.ts'

/** Does this path target the API's model-generation endpoint? Guards the
 *  orchestrator from rerouting non-generation calls (e.g. GET /v1/models). */
export function isGenerationPath(api: ModelApi, path: string): boolean {
  if (api === 'anthropic-messages') return path.endsWith('/messages')
  if (api === 'openai-chat') return path.endsWith('/chat/completions')
  return path.endsWith('/responses')
}

/** The canonical generation endpoint for a wire API, appended to a provider
 *  base URL. Provider base URLs follow the OpenAI/Anthropic SDK convention of
 *  ending at `/v1`; the version segment is only added when it is absent.
 *
 *  Exception: codex's ChatGPT session backend (`chatgpt.com/backend-api/codex`)
 *  serves `/responses` directly — no `/v1` segment. Wrapping that base with
 *  `/v1/responses` lands on a 403 HTML login page. Codex's own native flow
 *  hits the right path because it constructs it itself; when an off-agent
 *  route (openclaw, hermes, …) sends through this provider via agentfw, we
 *  rebuild the URL here and must match codex's convention. */
export function generationUrl(baseUrl: string, api: ModelApi): string {
  const base = baseUrl.replace(/\/+$/, '')
  const rest =
    api === 'anthropic-messages'
      ? 'messages'
      : api === 'openai-chat'
        ? 'chat/completions'
        : 'responses'
  if (isCodexChatGptBackend(base)) return `${base}/${rest}`
  return base.endsWith('/v1') ? `${base}/${rest}` : `${base}/v1/${rest}`
}

/** Rewrite the `model` field of a request body. All three wire formats carry
 *  the target model in a top-level `model` string, so one rewrite covers all.
 *  An unparseable body is forwarded unchanged. */
export function rewriteModel(body: ArrayBuffer, modelId: string): ArrayBuffer {
  try {
    const json: unknown = JSON.parse(new TextDecoder().decode(body))
    if (json && typeof json === 'object' && !Array.isArray(json)) {
      ;(json as Record<string, unknown>).model = modelId
      return encodeJson(json)
    }
  } catch {
    // Unparseable body — forward as-is; the model swap simply doesn't apply.
  }
  return body
}

/** True if this provider's base URL points at api.anthropic.com — the
 *  only place where Anthropic's server tools (web_search_20250305,
 *  code_execution_*, computer_*, etc.) actually execute. Used to gate
 *  server-tool stripping: if we're forwarding elsewhere (DeepSeek's
 *  /anthropic endpoint, an OpenAI-compatible provider, …), those tools
 *  have no executor on the other side and the model just emits broken
 *  tool_use calls. */
export function isAnthropicNative(provider: ProviderEntry): boolean {
  try {
    const host = new URL(provider.baseUrl).hostname
    return host === 'api.anthropic.com' || host.endsWith('.anthropic.com')
  } catch {
    return false
  }
}

/** Return a copy of the request body with Anthropic server tools removed
 *  from the top-level `tools` array. Server tools are identified by a
 *  `type` field that isn't the literal `'custom'` (web_search_20250305,
 *  code_execution_20250825, computer_*, bash_*, text_editor_*, …). A
 *  regular custom tool has only `name` + `input_schema` (no `type`, or
 *  `type:'custom'`) and is left untouched. MCP-namespaced tool names
 *  (`mcp__server__tool`) are custom tools too — they pass through.
 *
 *  Returns the same buffer when nothing was stripped (cheap fast path
 *  for the common case where the client sent no server tools). */
export function stripAnthropicServerToolsFromBody(body: ArrayBuffer): ArrayBuffer {
  let parsed: unknown
  try {
    parsed = JSON.parse(new TextDecoder().decode(body))
  } catch {
    return body
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return body
  const obj = parsed as Record<string, unknown>
  const filtered = withoutServerTools(obj)
  if (filtered === obj) return body
  return encodeJson(filtered)
}

/** In-memory variant — same filter, applied to a parsed request body.
 *  Returns the same reference when nothing was stripped; otherwise a
 *  shallow copy with `tools` replaced (or removed if empty). */
export function withoutServerTools(body: Record<string, unknown>): Record<string, unknown> {
  const tools = body.tools
  if (!Array.isArray(tools) || tools.length === 0) return body
  const kept = tools.filter((t) => {
    if (typeof t !== 'object' || t === null) return true
    const type = (t as { type?: unknown }).type
    if (typeof type !== 'string') return true
    return type === 'custom'
  })
  if (kept.length === tools.length) return body
  const next: Record<string, unknown> = { ...body }
  if (kept.length === 0) delete next.tools
  else next.tools = kept
  return next
}

/** Apply an auth spec to forwarded request headers. Passthrough auth
 *  leaves the client's own headers intact; otherwise every auth header the
 *  client may have sent is dropped before the configured one is injected.
 *  `agent-oauth` providers inject a subscription token agentfw reads — and
 *  co-refreshes — from the owning agent's own credential store. `id` is
 *  used purely for log labelling (provider id or route key). */
export async function applyAuth(headers: Headers, auth: ProviderAuth, id: string): Promise<void> {
  if (auth.kind === 'passthrough') return

  if (auth.kind === 'agent-oauth') {
    try {
      const tok = await getAgentToken(auth.agent)
      stripClientAuth(headers)
      headers.set('authorization', `Bearer ${tok.token}`)
      if (auth.agent === 'codex') {
        if (tok.accountId) headers.set('chatgpt-account-id', tok.accountId)
        // chatgpt.com/backend-api/codex is a Codex-CLI session endpoint —
        // not a generic OpenAI API. It checks the `originator` header for
        // a first-party Codex identity and rejects everything else with
        // a 403 HTML login page. The User-Agent must also claim Codex CLI
        // shape (the `codex_cli_rs/<ver>` prefix is the part that
        // matters; the rest is hand-waved). Setting both lets a non-codex
        // agent (hermes / openclaw) route through the same subscription.
        headers.set('originator', 'codex_cli_rs')
        headers.set('user-agent', CODEX_CLIENT_UA)
      }
      if (auth.agent === 'claude-code') {
        // Anthropic only accepts a Claude.ai subscription (OAuth) token on
        // the Messages API when the request claims Claude Code shape:
        //   • claude-code-20250219 — the Claude-Code-flavored beta gate
        //   • oauth-2025-04-20    — the OAuth-token beta gate
        // Both are required; without claude-code-20250219 the upstream
        // throttles hard (529 / aggressive rate-limit) even when auth itself
        // succeeds. Pattern matches openclaw's anthropic-transport-stream.
        const CC_OAUTH_BETAS = 'claude-code-20250219,oauth-2025-04-20'
        const beta = headers.get('anthropic-beta')
        headers.set('anthropic-beta', beta ? `${beta},${CC_OAUTH_BETAS}` : CC_OAUTH_BETAS)
        // Claude Code CLI's User-Agent is checked by the throttle layer.
        // Stamp it consistently so non-claude-code agents routing through
        // a Claude.ai subscription don't hit "unknown client" rate caps.
        headers.set('user-agent', CLAUDE_CODE_CLIENT_UA)
      }
    } catch (err) {
      // Token unavailable (locked Keychain, not logged in) — leave the
      // client's own auth in place and let it through as passthrough
      // rather than sending an unauthenticated request.
      logger.warn(
        `routing: ${auth.agent} OAuth unavailable for ${id}; ` +
          `passing the client's own auth through — ${(err as Error).message}`,
      )
    }
    return
  }

  stripClientAuth(headers)
  const value = getSecret(getSecrets(), auth.valueRef)
  if (!value) {
    logger.warn(`routing: secret "${auth.valueRef}" not set for ${id}`)
    return
  }
  if (auth.kind === 'bearer') {
    headers.set('authorization', `Bearer ${value}`)
  } else {
    headers.set(auth.header, value)
  }
}

/** Remove request-scoped Anthropic betas that don't survive a model swap. The
 *  1M long-context beta (`context-1m-*`) is tied to the client's original
 *  (1M-capable) model and tier; forwarding it to a routed member — a smaller
 *  companion, a different provider, or a Claude.ai-subscription call — earns a
 *  hard 400 ("The long context beta is not yet available for this
 *  subscription"). Other betas (prompt caching, tool betas) are broadly
 *  compatible and kept. No-op when the header is absent. */
export function dropSwapIncompatibleBetas(headers: Headers): void {
  const beta = headers.get('anthropic-beta')
  if (!beta) return
  const kept = beta
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s !== '' && !s.startsWith('context-1m'))
  if (kept.length > 0) headers.set('anthropic-beta', kept.join(','))
  else headers.delete('anthropic-beta')
}

/** Drop every auth header the client may have sent — used before injecting
 *  agentfw-managed credentials so the wrong agent's key can't leak upstream. */
function stripClientAuth(headers: Headers): void {
  headers.delete('authorization')
  headers.delete('x-api-key')
  headers.delete('api-key')
}

// User-Agent strings the upstream throttle / gate checks for first-party
// CLI shape. The `<originator>/<version>` prefix is what matters; the
// suffix mirrors codex/claude-code's own format so the request looks
// reasonable in their server logs. Bumping versions is harmless — the
// upstream only checks the prefix and (loosely) the shape.
const CODEX_CLIENT_UA = 'codex_cli_rs/0.81.0 (agentfw; openguardrails.com)'
const CLAUDE_CODE_CLIENT_UA = 'claude-cli/2.0.0 (agentfw; openguardrails.com)'

function encodeJson(value: unknown): ArrayBuffer {
  const bytes = new TextEncoder().encode(JSON.stringify(value))
  const out = new ArrayBuffer(bytes.byteLength)
  new Uint8Array(out).set(bytes)
  return out
}

/** A wire API → the decoder kind, used for pricing lookup and the captured
 *  packet's `protocol` field. */
export function apiToDecoder(api: ModelApi): DecoderKind {
  if (api === 'anthropic-messages') return 'anthropic'
  if (api === 'openai-chat') return 'openai-chat'
  return 'openai-responses'
}

// ── output-budget clamping ────────────────────────────────────────
//
// An agent sizes its output-token budget (`max_tokens`) for the provider it
// thinks it's talking to — Claude Code asks for up to 32 000 output tokens
// against Anthropic's 200 k window. When agentfw routes that request to a
// smaller third-party upstream (e.g. a 131 072-token vLLM model), the upstream
// rejects it the moment `input_tokens + max_tokens` crosses its window. The
// clamp shrinks the requested output budget so it fits the routed model's real
// limits, declared in models.json as `contextWindow` / `maxTokens`.
//
// Two rules keep it safe: we only ever *shrink* an existing budget (never
// inflate one, never impose a budget the client didn't ask for), and the
// input-token estimate only steers the clamp — a genuine input-only overflow is
// surfaced from the upstream's own (tokenizer-exact) error, not predicted here.

/** Deliberately small chars-per-token ratio so the estimate over-counts input
 *  tokens — over-counting only makes the clamp a little more conservative
 *  (shorter output), whereas under-counting could let the request graze the
 *  ceiling. Real text runs ~4 chars/token; JSON/code denser. */
const CHARS_PER_TOKEN = 3.5
/** Headroom held back from the window for tokenizer drift, chat-template
 *  markers, and special tokens the char estimate can't see. */
const WINDOW_SAFETY_MARGIN = 1024
/** Never clamp the output budget below this — a tiny budget is useless, and if
 *  the input genuinely leaves no room the upstream error is the right signal. */
const MIN_OUTPUT_TOKENS = 1024

/** The output-token budget field(s) for each wire API. OpenAI Chat accepts
 *  both the legacy `max_tokens` and the newer `max_completion_tokens`. */
function outputTokenFields(api: ModelApi): string[] {
  if (api === 'openai-responses') return ['max_output_tokens']
  if (api === 'openai-chat') return ['max_tokens', 'max_completion_tokens']
  return ['max_tokens']
}

function estimateInputTokens(body: Record<string, unknown>): number {
  return Math.ceil(JSON.stringify(body).length / CHARS_PER_TOKEN)
}

/** Shrink the request's output-token budget in place so it fits the routed
 *  model's `contextWindow` / `maxTokens`. No-op when neither is configured or
 *  the existing budget already fits. Returns true when it changed the body. */
export function clampOutputBudget(
  body: Record<string, unknown>,
  api: ModelApi,
  model: { id: string; contextWindow?: number; maxTokens?: number },
): boolean {
  const { contextWindow, maxTokens } = model
  if (contextWindow == null && maxTokens == null) return false

  let ceiling = Number.POSITIVE_INFINITY
  if (maxTokens != null) ceiling = maxTokens
  if (contextWindow != null) {
    const room = contextWindow - estimateInputTokens(body) - WINDOW_SAFETY_MARGIN
    ceiling = Math.min(ceiling, Math.max(MIN_OUTPUT_TOKENS, room))
  }
  if (!Number.isFinite(ceiling)) return false

  const cap = Math.floor(ceiling)
  let changed = false
  for (const field of outputTokenFields(api)) {
    const cur = body[field]
    if (typeof cur === 'number' && cur > cap) {
      body[field] = cap
      changed = true
      logger.info(
        `routing: clamped ${field} ${cur} → ${cap} for ${model.id} ` +
          `(contextWindow=${contextWindow ?? 'n/a'}, maxTokens=${maxTokens ?? 'n/a'})`,
      )
    }
  }
  return changed
}

/** Byte-level variant for the same-protocol fast path, which never parses the
 *  body. Returns the original buffer unchanged when there's nothing to clamp. */
export function clampOutputBudgetBytes(
  body: ArrayBuffer,
  api: ModelApi,
  model: { id: string; contextWindow?: number; maxTokens?: number },
): ArrayBuffer {
  if (model.contextWindow == null && model.maxTokens == null) return body
  let parsed: unknown
  try {
    parsed = JSON.parse(new TextDecoder().decode(body))
  } catch {
    return body
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return body
  const obj = parsed as Record<string, unknown>
  return clampOutputBudget(obj, api, model) ? encodeJson(obj) : body
}

/** Recognized upstream context-overflow error signatures (lower-cased). Covers
 *  vLLM ("maximum context length is N tokens … reduce the length of the input"),
 *  OpenAI-compatible servers (`context_length_exceeded`), and Anthropic. */
const OVERFLOW_SIGNATURES = [
  'maximum context length',
  'context length is',
  'reduce the length of the input',
  'context_length_exceeded',
  'too many tokens',
  'prompt is too long',
]

/** Turn an upstream error body into the text agentfw surfaces. A context-window
 *  overflow (the upstream's tokenizer-exact verdict that even the clamped
 *  request is too long) becomes an actionable message; everything else is just
 *  the trimmed excerpt. */
export function clarifyUpstreamError(status: number, text: string): string {
  const excerpt = errorExcerpt(text)
  if (status === 400 || status === 413 || status === 422) {
    const low = excerpt.toLowerCase()
    if (OVERFLOW_SIGNATURES.some((s) => low.includes(s))) {
      const guidance =
        'agentfw: context window exceeded — the conversation is longer than the ' +
        "routed model's context window. Run /compact or /clear in the agent, " +
        'lower CLAUDE_CODE_MAX_OUTPUT_TOKENS, or route to a larger-context model. '
      return `${guidance}Upstream said: ${excerpt}`
    }
  }
  return excerpt || `HTTP ${status}`
}

/** Headroom held back from the window on a forced retry. Smaller than the
 *  estimate-based margin because here the input count is the upstream's own
 *  exact figure, not a guess. */
const RETRY_SAFETY_MARGIN = 256

/** Parse the model's context window and the prompt's input-token count out of
 *  an upstream overflow error. vLLM ("maximum context length is N tokens …
 *  contains at least M input tokens") and OpenAI-compatible servers ("…
 *  resulted in M tokens") both state these. */
export function parseOverflowNumbers(text: string): { window?: number; input?: number } {
  const window = /maximum context length is (\d+)/i.exec(text)?.[1]
  const input =
    /contains at least (\d+) input tokens/i.exec(text)?.[1] ??
    /(\d+) input tokens/i.exec(text)?.[1] ??
    /resulted in (\d+) tokens/i.exec(text)?.[1]
  return {
    ...(window ? { window: Number(window) } : {}),
    ...(input ? { input: Number(input) } : {}),
  }
}

/** From a context-overflow error, compute an output-token budget that fits the
 *  window using the upstream's exact reported numbers — so the request can be
 *  retried once and succeed. Undefined when the numbers aren't present or the
 *  input alone leaves no usable room (a genuine input-only overflow). The
 *  configured `contextWindow` wins over the parsed one when both exist. */
export function safeRetryBudget(
  text: string,
  model: { contextWindow?: number },
): number | undefined {
  const parsed = parseOverflowNumbers(text)
  const window = model.contextWindow ?? parsed.window
  if (window == null || parsed.input == null) return undefined
  const budget = window - parsed.input - RETRY_SAFETY_MARGIN
  return budget >= RETRY_SAFETY_MARGIN ? budget : undefined
}

function isOverflowStatus(status: number): boolean {
  return status === 400 || status === 413 || status === 422
}

/** The outcome of one buffered upstream call. */
export type AttemptResult =
  | {
      ok: true
      status: number
      json: unknown
      ir: IRResponse
      durMs: number
      startedAtWall: number
      upstreamUrl: string
    }
  | {
      ok: false
      status: number
      errorText: string
      /** Raw upstream response headers captured on the error path. Useful for
       *  diagnosing 400/401/etc. where the body alone (often empty or
       *  unhelpfully terse) doesn't say what went wrong. */
      errorHeaders?: Record<string, string>
      durMs: number
      startedAtWall: number
      upstreamUrl: string
    }

/** A built upstream request plus the fake→real map for restoring its response
 *  (empty when masking is off for the member's provider). */
type BuiltRequest = { request: Request; restore: Map<string, string> }

/** Build the upstream HTTP request for a routed member: translate the client
 *  request into the member's wire format, force the model and the stream flag,
 *  apply provider auth, mask credentials for the member's provider, and target
 *  the member API's generation endpoint. */
async function buildUpstreamRequest(
  member: ResolvedMember,
  clientApi: ModelApi,
  clientRequest: Record<string, unknown>,
  ctx: { agent: AgentId; reqHeaders: Headers },
  upstreamUrl: string,
  stream: boolean,
  /** Hard cap on the output-token budget, set on a context-overflow retry to
   *  the exact figure the upstream's error implies. */
  outputOverride?: number,
): Promise<BuiltRequest> {
  // Strip Anthropic server tools before translation when the destination
  // can't execute them. Anthropic's web_search / code_execution / etc.
  // only work when the API call lands on api.anthropic.com; forwarding
  // them to a third-party (even an Anthropic-API-compatible upstream
  // like DeepSeek's /anthropic) lets the model emit tool_use calls that
  // never get results. Filtering at the source keeps the IR clean and
  // prevents the bogus tool from showing up after translation either.
  const sourceRequest = isAnthropicNative(member.provider)
    ? clientRequest
    : withoutServerTools(clientRequest)
  const translated = translateRequest(clientApi, member.api, sourceRequest) as Record<
    string,
    unknown
  >
  const upstreamBody: Record<string, unknown> = {
    ...translated,
    model: member.model.id,
    stream,
  }

  // Claude.ai subscription routes need both the beta flags (set in
  // applyAuth) and an exact-match `system` field — the adapter rewrites
  // the body to match what Claude Code CLI sends.
  if (
    member.api === 'anthropic-messages' &&
    member.provider.auth.kind === 'agent-oauth' &&
    member.provider.auth.agent === 'claude-code'
  ) {
    adaptForClaudeCodeOAuth(upstreamBody, member.provider.reasoningEffort)
  }

  // codex's ChatGPT session endpoint speaks a session-bound subset of the
  // public Responses API; the body adapter mirrors codex-CLI's own request
  // shape (store, reasoning.effort, include, text.verbosity, …).
  if (member.api === 'openai-responses' && isCodexChatGptBackend(member.provider.baseUrl)) {
    adaptForCodexBackend(upstreamBody, member.provider.reasoningEffort)
  }

  // Shrink the output-token budget to fit the routed model's declared
  // context window / max output, so a request the agent sized for a larger
  // window doesn't get rejected by a smaller third-party upstream.
  clampOutputBudget(upstreamBody, member.api, member.model)
  if (outputOverride != null) {
    const fields = outputTokenFields(member.api)
    let set = false
    for (const field of fields) {
      if (typeof upstreamBody[field] === 'number') {
        upstreamBody[field] = Math.min(upstreamBody[field] as number, outputOverride)
        set = true
      }
    }
    // No budget field present (rare) — set the API's primary one so the cap
    // actually takes effect on retry.
    if (!set && fields[0]) upstreamBody[fields[0]] = outputOverride
  }

  const headers = filterRequestHeaders(ctx.reqHeaders, new URL(upstreamUrl).host)
  dropSwapIncompatibleBetas(headers)
  headers.set('content-type', 'application/json')
  headers.set('accept', stream ? 'text/event-stream' : 'application/json')
  if (member.api === 'anthropic-messages' && !headers.has('anthropic-version')) {
    headers.set('anthropic-version', '2023-06-01')
  }
  if (member.provider.auth.kind === 'passthrough') {
    const extra = await dynamicHeadersFor(ctx.agent)
    for (const [k, v] of Object.entries(extra)) headers.set(k, v)
  } else {
    await applyAuth(headers, member.provider.auth, member.provider.id)
  }

  // Credential masking: swap real secrets in the final upstream body for fixed
  // fakes before it leaves the machine, scoped to this provider. The response is
  // un-masked in execAttempt / execStream below.
  const bodyText = JSON.stringify(upstreamBody)
  const masked = maskText(bodyText, member.provider.id)

  return {
    request: new Request(upstreamUrl, {
      method: 'POST',
      headers,
      body: masked?.text ?? bodyText,
    }),
    restore: masked?.restore ?? new Map<string, string>(),
  }
}

/** Wrap a streamed upstream body so masked fakes are restored before the bytes
 *  reach the caller (and, via the tee, the client). No-op when nothing was
 *  masked. */
function restoreStream(
  body: ReadableStream<Uint8Array>,
  restore: Map<string, string>,
): ReadableStream<Uint8Array> {
  return restore.size > 0 ? body.pipeThrough(makeRestoreTransform(restore)) : body
}

/** Run one buffered upstream call against a resolved member: translate the
 *  client request into the member's wire format, force the model and
 *  `stream:false`, apply auth, fetch, and parse the response into IR. Never
 *  throws — a network error or a non-2xx status becomes an `ok:false` result
 *  the orchestrator fails over from. */
export async function execAttempt(
  member: ResolvedMember,
  clientApi: ModelApi,
  clientRequest: Record<string, unknown>,
  ctx: { agent: AgentId; reqHeaders: Headers },
): Promise<AttemptResult> {
  const startedAtWall = Date.now()
  const t0 = performance.now()
  const upstreamUrl = generationUrl(member.provider.baseUrl, member.api)

  try {
    const { request, restore } = await buildUpstreamRequest(
      member,
      clientApi,
      clientRequest,
      ctx,
      upstreamUrl,
      false,
    )
    let res = await fetch(request)
    let text = restoreText(await res.text(), restore)

    // Context-overflow retry: the upstream rejected because input + output
    // crossed its window. Its error reports the exact figures, so retry once
    // with an output budget that fits — turning a hard failure (often on the
    // very first turn, when there's nothing to compact) into a success. Only
    // the output budget shrinks; no message content is dropped.
    if (res.status >= 400 && isOverflowStatus(res.status)) {
      const budget = safeRetryBudget(text, member.model)
      if (budget != null) {
        const retry = await buildUpstreamRequest(
          member,
          clientApi,
          clientRequest,
          ctx,
          upstreamUrl,
          false,
          budget,
        )
        const res2 = await fetch(retry.request)
        const text2 = restoreText(await res2.text(), retry.restore)
        if (res2.status < 400) {
          logger.info(
            `routing: ${member.model.id} retried with max output ${budget} after context overflow`,
          )
          res = res2
          text = text2
        }
      }
    }

    const durMs = performance.now() - t0

    if (res.status >= 400) {
      return {
        ok: false,
        status: res.status,
        errorText: clarifyUpstreamError(res.status, text),
        errorHeaders: snapshotHeaders(res),
        durMs,
        startedAtWall,
        upstreamUrl,
      }
    }

    let json: unknown
    try {
      json = JSON.parse(text)
    } catch {
      return {
        ok: false,
        status: res.status,
        errorText: `non-JSON response: ${errorExcerpt(text)}`,
        errorHeaders: snapshotHeaders(res),
        durMs,
        startedAtWall,
        upstreamUrl,
      }
    }

    return {
      ok: true,
      status: res.status,
      json,
      ir: parseResponseToIR(member.api, json),
      durMs,
      startedAtWall,
      upstreamUrl,
    }
  } catch (err) {
    return {
      ok: false,
      status: 0,
      errorText: `upstream call failed: ${(err as Error).message}`,
      durMs: performance.now() - t0,
      startedAtWall,
      upstreamUrl,
    }
  }
}

/** The outcome of one streaming upstream call. A success hands back the live
 *  SSE body unread, for the caller to tee; a failure carries the read body. */
export type StreamAttempt =
  | {
      ok: true
      status: number
      body: ReadableStream<Uint8Array>
      durMs: number
      startedAtWall: number
      upstreamUrl: string
    }
  | {
      ok: false
      status: number
      errorText: string
      errorHeaders?: Record<string, string>
      durMs: number
      startedAtWall: number
      upstreamUrl: string
    }

/** Run one streaming upstream call against a resolved member: translate the
 *  client request, force the model and `stream:true`, apply auth, fetch, and
 *  return the live SSE body. Never throws — a network error or a non-2xx
 *  status becomes an `ok:false` result. */
export async function execStream(
  member: ResolvedMember,
  clientApi: ModelApi,
  clientRequest: Record<string, unknown>,
  ctx: { agent: AgentId; reqHeaders: Headers },
): Promise<StreamAttempt> {
  const startedAtWall = Date.now()
  const t0 = performance.now()
  const upstreamUrl = generationUrl(member.provider.baseUrl, member.api)

  try {
    const { request, restore } = await buildUpstreamRequest(
      member,
      clientApi,
      clientRequest,
      ctx,
      upstreamUrl,
      true,
    )
    const res = await fetch(request)

    if (res.status >= 400 && isOverflowStatus(res.status)) {
      // Same context-overflow retry as the buffered path — the error body must
      // be read here, which is fine since we only read it on the failure path.
      const text = restoreText(await res.text(), restore)
      const budget = safeRetryBudget(text, member.model)
      if (budget != null) {
        const retry = await buildUpstreamRequest(
          member,
          clientApi,
          clientRequest,
          ctx,
          upstreamUrl,
          true,
          budget,
        )
        const res2 = await fetch(retry.request)
        if (res2.status < 400 && res2.body) {
          logger.info(
            `routing: ${member.model.id} retried with max output ${budget} after context overflow [stream]`,
          )
          return {
            ok: true,
            status: res2.status,
            body: restoreStream(res2.body, retry.restore),
            durMs: performance.now() - t0,
            startedAtWall,
            upstreamUrl,
          }
        }
      }
      return {
        ok: false,
        status: res.status,
        errorText: clarifyUpstreamError(res.status, text),
        errorHeaders: snapshotHeaders(res),
        durMs: performance.now() - t0,
        startedAtWall,
        upstreamUrl,
      }
    }

    const durMs = performance.now() - t0

    if (res.status >= 400 || !res.body) {
      const text = res.body ? restoreText(await res.text(), restore) : ''
      return {
        ok: false,
        status: res.status,
        errorText: clarifyUpstreamError(res.status, text),
        errorHeaders: snapshotHeaders(res),
        durMs,
        startedAtWall,
        upstreamUrl,
      }
    }

    return {
      ok: true,
      status: res.status,
      body: restoreStream(res.body, restore),
      durMs,
      startedAtWall,
      upstreamUrl,
    }
  } catch (err) {
    return {
      ok: false,
      status: 0,
      errorText: `upstream call failed: ${(err as Error).message}`,
      durMs: performance.now() - t0,
      startedAtWall,
      upstreamUrl,
    }
  }
}

/** Trim an upstream error body to a short single-line excerpt — strips HTML
 *  tags and collapses whitespace, for logs and the captured `error` field. */
function errorExcerpt(text: string): string {
  return text
    .replace(/<[^>]+>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 200)
}

/** Snapshot an upstream Response's headers into a plain object, for the
 *  diagnostic `errorHeaders` field on a failed attempt. `set-cookie` is
 *  dropped — it would leak the upstream's session state and is never
 *  load-bearing for debugging a 4xx; values longer than 1 KB are truncated
 *  so a misbehaving header can't bloat the captured payload. */
function snapshotHeaders(res: Response): Record<string, string> {
  const out: Record<string, string> = {}
  res.headers.forEach((value, key) => {
    if (key.toLowerCase() === 'set-cookie') return
    out[key] = value.length > 1024 ? `${value.slice(0, 1024)}…` : value
  })
  return out
}
