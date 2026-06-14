// The routing orchestrator — the seam between buffering the request body and
// fetching upstream. When a route is configured to swap models, the
// orchestrator owns the upstream call; otherwise it returns undefined and the
// proxy falls through to plain passthrough (the byte-identical fast path).
//
// Three execution paths:
//   - same-protocol single-model swap → tee straight through, no translation
//     (`runSameProtocolSwap`, the Stage 3 fast path).
//   - cross-protocol single-model swap → buffer + translate + synthesize
//     (`runBuffered`), or true-stream the translation (`runStreamingSwap`).
//   - chain → buffer, walk members with failover (error / budget / token
//     rules), running the text↔multimodal vision loop for any text-only
//     member when the route carries a vision capability (`runChain`).

import type { AgentId } from '../../core/agent.ts'
import { logger } from '../../core/logger.ts'
import type { ModelApi } from '../../core/model-registry.ts'
import type { Orchestration, RiskTag } from '../../core/packet.ts'
import type { DecoderKind } from '../../core/routes.ts'
import { streamTranslationEnabled, type SwitchRule } from '../../core/routing-policy.ts'
import { decoderFor } from '../decoders/index.ts'
import { maskRequestBody, restoreResponseStream } from '../proxy/credential-mask.ts'
import { dynamicHeadersFor } from '../proxy/dynamic-headers.ts'
import { filterRequestHeaders, filterResponseHeaders } from '../proxy/forward.ts'
import { beginRequest, endRequest, trackStream } from '../proxy/inflight.ts'
import { getRoutingPolicy } from '../routing/load.ts'
import {
  type IRResponse,
  parseRequestToIR,
  parseStreamToIR,
  serializeRequestFromIR,
  serializeResponseFromIR,
  translateSseStream,
} from '../translate/index.ts'
import { type IRMessage, type IRRequest, mergeConsecutive } from '../translate/ir.ts'
import { spendInPeriod, tokensInPeriod } from './budget.ts'
import { captureChain, captureSingle, type RoutedAttempt } from './capture.ts'
import {
  type AttemptResult,
  applyAuth,
  clampOutputBudgetBytes,
  dropSwapIncompatibleBetas,
  execAttempt,
  execStream,
  generationUrl,
  isAnthropicNative,
  isGenerationPath,
  rewriteModel,
  stripAnthropicServerToolsFromBody,
} from './exec.ts'
import {
  hasAnthropicWebSearchTool,
  requestHasWebSearchServerTool,
  rewriteAnthropicWebSearchTool,
  runWebSearchEmulationLoop,
} from './web-search-emulation.ts'
import { preDescribeImages, requestHasImageBlock } from './image-predescribe.ts'
import {
  type ModelRef,
  type ResolvedMember,
  type ResolvedPanelMember,
  type ResolvedRoute,
  resolveRoute,
} from './resolve.ts'
import { resolveSubagentDowngrade } from './subagent.ts'
import { synthesizeSse } from './synth-sse.ts'

export type WireContext = {
  agent: AgentId
  provider: string
  /** Route-table key — `<agent>/<modelId>` for wrap-style, `<agent>/*`
   *  for wildcard. Carries capture identity (the seeded providerId). */
  routeKey: string
  /** Routing-policy lookup key — `<agent>/<bodyModel>` (or `<agent>/*`
   *  when no body model). Distinct from routeKey so wildcard routes can
   *  carry per-source-model overrides while keeping a single seeded
   *  provider for capture attribution. */
  policyKey: string
  /** Wire-derived agent instance — the `@<instance>` path segment. Folded
   *  into policyKey already; carried here too so capture attributes the
   *  routed call to the right instance. */
  instanceId?: string
  decoder: DecoderKind
  reqMethod: string
  /** Upstream-relative path, e.g. "/v1/messages". */
  restPath: string
  reqHeaders: Headers
  reqBody: ArrayBuffer | undefined
}

/** The routing seam. Returns a Response when the orchestrator owns the
 *  upstream call, or undefined to fall through to plain passthrough. */
export async function tryOrchestrate(ctx: WireContext): Promise<Response | undefined> {
  // Subagent cost-saver takes precedence: a Claude Code dynamic-workflow
  // subagent call is rerouted to the cheaper model regardless of the route's
  // normal policy (the planner is left untouched and falls through below).
  const resolved = resolveSubagentDowngrade(ctx) ?? resolveRoute(ctx.policyKey, ctx.decoder)
  if (resolved.kind === 'passthrough') return undefined
  // Only model-generation POSTs are reroutable — never GET /v1/models etc.
  if (ctx.reqMethod !== 'POST' || !ctx.reqBody) return undefined
  if (!isGenerationPath(resolved.clientApi, ctx.restPath)) return undefined

  // Same-protocol single-model swap — the byte-tee fast path, no translation
  // and no body parse. Skipped when a vision capability is configured on the
  // route OR when web_search emulation might fire (non-Anthropic target +
  // Anthropic-messages decoder). Both loop triggers need a parsed body, so
  // we slow-path through runBuffered which dispatches accordingly. The
  // body-shape probe for web_search is a cheap string scan so we don't
  // pay the parse cost just to decide whether to parse.
  if (
    resolved.kind === 'model' &&
    resolved.api === resolved.clientApi &&
    resolved.capabilities.vision == null &&
    !maybeNeedsWebSearchEmulation(ctx, resolved)
  ) {
    return runSameProtocolSwap(ctx, resolved, ctx.reqBody)
  }

  // Every other path needs the request as a parsed JSON object — a non-JSON
  // body is not reroutable and falls through to passthrough.
  const req = parseJsonObject(ctx.reqBody)
  if (!req) return undefined

  if (resolved.kind === 'fusion') {
    return runFusion(ctx, resolved, req)
  }

  if (resolved.kind === 'chain') {
    return runChain(ctx, resolved, req)
  }

  // A cross-protocol single-model swap. With stream:true → true-stream the
  // translation, unless the global escape hatch forces buffer + synthesize.
  if (req.stream === true && streamTranslationEnabled(getRoutingPolicy())) {
    return runStreamingSwap(ctx, resolved, req)
  }
  return runBuffered(ctx, resolved, req)
}

/** Cheap, parse-free probe: does this Anthropic-bound request likely
 *  carry the `web_search_20250305` server tool, AND is the target
 *  non-Anthropic? When both, the fast path is bypassed so runBuffered
 *  can swap the server tool for a custom tool and run the executor
 *  loop. False positives just cost one body parse — never wrong-route.
 *  False negatives would silently fall back to the strip behavior,
 *  which is acceptable for v1. */
function maybeNeedsWebSearchEmulation(
  ctx: WireContext,
  resolved: Extract<ResolvedRoute, { kind: 'model' }>,
): boolean {
  if (resolved.clientApi !== 'anthropic-messages') return false
  if (isAnthropicNative(resolved.provider)) return false
  if (!ctx.reqBody || ctx.reqBody.byteLength === 0) return false
  // The decoder is anthropic, so the body is JSON. A substring scan is
  // enough — the tool type is a unique, opaque string that's never
  // ambiguous with surrounding content.
  const bytes = new Uint8Array(ctx.reqBody)
  const needle = new TextEncoder().encode('web_search_20250305')
  return indexOfBytes(bytes, needle) >= 0
}

function indexOfBytes(haystack: Uint8Array, needle: Uint8Array): number {
  if (needle.length === 0 || needle.length > haystack.length) return -1
  outer: for (let i = 0; i <= haystack.length - needle.length; i++) {
    for (let j = 0; j < needle.length; j++) {
      if (haystack[i + j] !== needle[j]) continue outer
    }
    return i
  }
  return -1
}

/** Parse a request body into a plain JSON object, or undefined when it is not
 *  one — a non-JSON body is not reroutable and falls through to passthrough. */
function parseJsonObject(body: ArrayBuffer): Record<string, unknown> | undefined {
  try {
    const json: unknown = JSON.parse(new TextDecoder().decode(body))
    if (json && typeof json === 'object' && !Array.isArray(json)) {
      return json as Record<string, unknown>
    }
  } catch {
    // non-JSON body — not reroutable
  }
  return undefined
}

// ── buffered path: single cross-protocol swap ─────────────────────

/** Buffer a single cross-protocol model swap: one upstream call, the response
 *  translated into the client's wire format, captured as one packet. When the
 *  route carries a vision capability and the request has images the model
 *  can't see, the same primary member runs the text↔multimodal vision loop
 *  in place of a plain attempt — same machinery as the chain path. */
async function runBuffered(
  ctx: WireContext,
  resolved: Extract<ResolvedRoute, { kind: 'model' }>,
  req: Record<string, unknown>,
): Promise<Response> {
  const wantsStream = req.stream === true
  const member: ResolvedMember = {
    model: resolved.model,
    provider: resolved.provider,
    api: resolved.api,
    switchOn: [],
  }
  const execCtx = { agent: ctx.agent, reqHeaders: ctx.reqHeaders }
  const orchStartWall = Date.now()
  const orchT0 = performance.now()

  beginRequest()
  try {
    // Web-search tool emulation: only fires when the route has an
    // explicit per-route provider pin
    // (capabilities.web_search = {via:'local', providerId:'X'}).
    // No more silent global-active fallback — "agent handles
    // natively" is the safe default per route, and the user makes a
    // deliberate per-route choice to hand search to agentfw. For
    // Anthropic routes the agent's own server tool still works
    // natively; for non-Anthropic routes without a pin, the server
    // tool gets stripped (existing behaviour) and the model knows it
    // has no search.
    const wsCap = resolved.capabilities.web_search
    const shouldEmulateSearch =
      resolved.clientApi === 'anthropic-messages' &&
      !isAnthropicNative(resolved.provider) &&
      hasAnthropicWebSearchTool(req) &&
      wsCap?.via === 'local' &&
      typeof wsCap.providerId === 'string' &&
      wsCap.providerId !== ''

    // Pre-describe pass: when the request carries image blocks but the routed
    // model can't see them, run the route's configured vision companion first
    // to convert each image into a text description and forward a text-only
    // request. agentfw never auto-picks a companion — an image bound for a
    // text-only model with no companion configured is a misconfiguration, so
    // we fail with an actionable message instead of guessing a vision model
    // (which surfaces confusing, unrelated upstream errors) or dropping it.
    const visionCompanion = resolved.capabilities.vision
    const targetIsTextOnly = !(resolved.model.input ?? []).includes('image')
    const hasImage = requestHasImageBlock(resolved.clientApi, req)

    if (hasImage && targetIsTextOnly && visionCompanion?.via !== 'companion') {
      logger.warn(
        `routing: ${ctx.routeKey} — image present but ${resolved.model.id} is text-only and no ` +
          'vision companion is configured for this route',
      )
      return new Response(
        JSON.stringify({
          error:
            `agentfw: routed model "${resolved.model.id}" can't see images and no vision companion ` +
            "is configured for this route. Configure a vision companion (the route's Vision " +
            'setting, or `agentfw route vision <route> --companion <model>`) so agentfw can ' +
            'describe images for it.',
        }),
        { status: 400, headers: { 'content-type': 'application/json' } },
      )
    }

    let workingReq = req
    let preDescribeAttempts: RoutedAttempt[] = []
    let visionMemberToUse: ResolvedMember | undefined
    if (visionCompanion?.via === 'companion' && hasImage) {
      visionMemberToUse = {
        model: visionCompanion.ref.model,
        provider: visionCompanion.ref.provider,
        api: visionCompanion.ref.api,
        switchOn: [],
      }
    }
    if (visionMemberToUse) {
      const pre = await preDescribeImages(resolved.clientApi, req, visionMemberToUse, execCtx)
      preDescribeAttempts = pre.attempts
      workingReq = pre.request
      logger.info(
        `routed ${ctx.routeKey} → ${member.model.id} [pre-describe ` +
          `${pre.attempts.length} image(s) via ${visionMemberToUse.model.id}, ok=${pre.ok}]`,
      )
    }

    if (shouldEmulateSearch) {
      const rewritten = rewriteAnthropicWebSearchTool(workingReq)
      // wsCap guaranteed to be {via:'local', providerId} by the
      // shouldEmulateSearch predicate above.
      const providerIdOverride =
        wsCap?.via === 'local' && wsCap.providerId ? wsCap.providerId : undefined
      const outcome = await runWebSearchEmulationLoop({
        member,
        clientApi: resolved.clientApi,
        clientRequest: rewritten,
        ctx: execCtx,
        ...(providerIdOverride ? { providerIdOverride } : {}),
      })
      if (outcome.ok) {
        logger.info(
          `routed ${ctx.routeKey} → ${member.model.id} (${member.provider.id}) ` +
            `[web_search emulation, ${outcome.searches.size} search(es)]`,
        )
      } else {
        logger.warn(`routing: ${ctx.routeKey} → ${member.model.id} web_search emulation failed`)
      }
      // The final response is the model's last assistant turn, returned
      // verbatim in the upstream JSON shape (Anthropic messages). v1
      // does not synthesize server_tool_use / web_search_tool_result
      // blocks — the agent's user-visible content (the final text) is
      // intact; Claude Code's wrapper-tool parser (which counts these
      // blocks) is a v2 concern.
      const last = outcome.attempts[outcome.attempts.length - 1]
      const winnerAttempt: RoutedAttempt | undefined = outcome.ok && last ? last : undefined
      // Vision pre-describe + web-search emulation attempts under one
      // captured parent. Renumber steps in append order so the dashboard
      // shows them as siblings.
      const allAttempts: RoutedAttempt[] = [...preDescribeAttempts, ...outcome.attempts].map(
        (a, i) => ({ ...a, step: i }),
      )
      const response = buildChainResponse(
        resolved.clientApi,
        winnerAttempt?.result.ok
          ? { ir: winnerAttempt.result.ir, status: outcome.status }
          : undefined,
        allAttempts,
        wantsStream,
      )
      await captureChain(
        {
          agent: ctx.agent,
          decoder: ctx.decoder,
          ...(ctx.instanceId ? { instanceId: ctx.instanceId } : {}),
        },
        resolved.clientApi,
        workingReq,
        allAttempts,
        winnerAttempt?.result.ok
          ? { ir: winnerAttempt.result.ir, status: outcome.status }
          : undefined,
        resolved.configuredTarget,
        { ts: orchStartWall, durMs: performance.now() - orchT0 },
        [],
        outcome.toolExecutions,
      )
      return response
    }

    const result = await execAttempt(member, resolved.clientApi, workingReq, execCtx)
    if (result.ok) {
      logger.info(`routed ${ctx.routeKey} → ${member.model.id} (${member.provider.id})`)
    } else {
      logger.warn(
        `routing: ${ctx.routeKey} → ${member.model.id} failed ` +
          `(${result.status}): ${result.errorText}`,
      )
    }

    const primary: RoutedAttempt = { member, result, role: 'primary', step: 0 }
    // When pre-describe attempts ran, the routed call is part of a
    // multi-attempt run and needs the same capture shape the chain path
    // uses so the dashboard surfaces the vision-companion calls under
    // the same parent. Otherwise fall through to the cheap single capture.
    if (preDescribeAttempts.length > 0) {
      const allAttempts: RoutedAttempt[] = [
        ...preDescribeAttempts,
        { ...primary, step: preDescribeAttempts.length },
      ].map((a, i) => ({ ...a, step: i }))
      const response = buildChainResponse(
        resolved.clientApi,
        result.ok ? { ir: result.ir, status: result.status } : undefined,
        allAttempts,
        wantsStream,
      )
      await captureChain(
        {
          agent: ctx.agent,
          decoder: ctx.decoder,
          ...(ctx.instanceId ? { instanceId: ctx.instanceId } : {}),
        },
        resolved.clientApi,
        workingReq,
        allAttempts,
        result.ok ? { ir: result.ir, status: result.status } : undefined,
        resolved.configuredTarget,
        { ts: orchStartWall, durMs: performance.now() - orchT0 },
        [],
      )
      return response
    }
    const response = buildClientResponse(
      resolved.clientApi,
      result.ok ? primary : undefined,
      [primary],
      wantsStream,
    )
    await captureSingle(
      {
        agent: ctx.agent,
        decoder: ctx.decoder,
        ...(ctx.instanceId ? { instanceId: ctx.instanceId } : {}),
      },
      resolved.clientApi,
      workingReq,
      primary,
      resolved.configuredTarget,
    )
    return response
  } finally {
    endRequest()
  }
}

/** Resolve a route's per-route vision companion into the model/provider/api
 *  the chain path's pre-describe pass calls. Returns undefined when there is
 *  no companion configured (caller short-circuits without parsing the request
 *  body to look for images). */
function visionCompanionShim(
  resolved: Extract<ResolvedRoute, { kind: 'model' } | { kind: 'chain' }>,
):
  | {
      kind: 'vision'
      model: ResolvedMember['model']
      provider: ResolvedMember['provider']
      api: ModelApi
    }
  | undefined {
  const cap = resolved.capabilities.vision
  if (cap?.via !== 'companion') return undefined
  return { kind: 'vision', model: cap.ref.model, provider: cap.ref.provider, api: cap.ref.api }
}

/** Render a single attempt as a client response, or an error JSON when it
 *  failed. A `stream:true` client gets a synthesized SSE body (the buffered
 *  path cannot true-stream — see synth-sse.ts). */
function buildClientResponse(
  clientApi: ModelApi,
  winner: RoutedAttempt | undefined,
  attempts: RoutedAttempt[],
  wantsStream: boolean,
): Response {
  if (winner && winner.result.ok) {
    const { ir, json, status } = winner.result
    if (wantsStream) {
      return new Response(synthesizeSse(clientApi, ir), {
        status,
        headers: { 'content-type': 'text/event-stream' },
      })
    }
    // Same wire format → return the upstream JSON untouched; cross-protocol →
    // re-serialize the IR into the client's format.
    const body = winner.member.api === clientApi ? json : serializeResponseFromIR(clientApi, ir)
    return new Response(JSON.stringify(body), {
      status,
      headers: { 'content-type': 'application/json' },
    })
  }

  const last = attempts[attempts.length - 1]
  const status = last && !last.result.ok && last.result.status >= 400 ? last.result.status : 502
  const detail = last && !last.result.ok ? last.result.errorText : 'no upstream attempt completed'
  return new Response(JSON.stringify({ error: 'agentfw: routed upstream call failed', detail }), {
    status,
    headers: { 'content-type': 'application/json' },
  })
}

// ── chain path: failover walk + per-member vision loop ──────────

/** Walk a chain's members, failing over and budget/token-switching as
 *  configured. Any text-only member runs the text↔multimodal loop in place
 *  of a plain attempt when the route carries a vision capability. The
 *  response is built from the winning member; capture records the whole
 *  fan-out afterwards so the client is never blocked on it. */
async function runChain(
  ctx: WireContext,
  resolved: Extract<ResolvedRoute, { kind: 'chain' }>,
  req: Record<string, unknown>,
): Promise<Response> {
  const orchStartWall = Date.now()
  const orchT0 = performance.now()
  const wantsStream = req.stream === true
  const { clientApi, members } = resolved
  const visionToolModel = visionCompanionShim(resolved)
  const execCtx = { agent: ctx.agent, reqHeaders: ctx.reqHeaders }

  beginRequest()
  try {
    const attempts: RoutedAttempt[] = []
    const risk: RiskTag[] = []
    let winner: { ir: IRResponse; status: number } | undefined
    let pendingRole: Orchestration['role'] | undefined
    let step = 0

    // Pre-describe images up front, once for the whole chain. A
    // text-only failover member (e.g. MiniMax-M2.7) would otherwise
    // 400 the request the moment we switched to it. The cached
    // descriptions get reused across members so we don't pay the
    // companion call twice. Mirrors runBuffered's pre-describe block —
    // the image cache shares per-image work between the two paths.
    let workingReq = req
    const hasImage = requestHasImageBlock(clientApi, req)
    if (visionToolModel && hasImage) {
      const visionMember: ResolvedMember = {
        model: visionToolModel.model,
        provider: visionToolModel.provider,
        api: visionToolModel.api,
        switchOn: [],
      }
      const pre = await preDescribeImages(clientApi, req, visionMember, execCtx)
      for (const a of pre.attempts) attempts.push({ ...a, step: step++ })
      workingReq = pre.request
      logger.info(
        `routed ${ctx.routeKey} [pre-describe ${pre.attempts.length} image(s) ` +
          `via ${visionToolModel.model.id}, ok=${pre.ok}]`,
      )
    }

    for (let i = 0; i < members.length; i++) {
      const member = members[i]
      if (!member) continue
      const isLast = i === members.length - 1

      // Quota pre-checks — only meaningful when a fallback remains to switch
      // to. Budget (USD) and tokens are independent caps; either trips
      // a switch. The last member's rules are ignored unconditionally.
      if (!isLast) {
        const budgetRule = member.switchOn.find(
          (r): r is Extract<SwitchRule, { kind: 'budget' }> => r.kind === 'budget',
        )
        if (budgetRule) {
          const spent = await spendInPeriod(member.model.id, budgetRule.period)
          if (spent >= budgetRule.usdLimit) {
            logger.info(
              `routing: ${ctx.routeKey} member ${member.model.id} over budget ` +
                `($${spent.toFixed(2)} ≥ $${budgetRule.usdLimit}/${budgetRule.period}), switching`,
            )
            pendingRole = 'budget-switch'
            continue
          }
        }
        const tokenRule = member.switchOn.find(
          (r): r is Extract<SwitchRule, { kind: 'tokens' }> => r.kind === 'tokens',
        )
        if (tokenRule) {
          const used = await tokensInPeriod(member.model.id, tokenRule.period)
          if (used >= tokenRule.tokenLimit) {
            logger.info(
              `routing: ${ctx.routeKey} member ${member.model.id} over token quota ` +
                `(${used} ≥ ${tokenRule.tokenLimit}/${tokenRule.period}), switching`,
            )
            pendingRole = 'budget-switch'
            continue
          }
        }
      }

      const role: Orchestration['role'] =
        pendingRole ?? (attempts.length === 0 ? 'primary' : 'failover')
      pendingRole = undefined

      // Per-member web-search emulation: same predicate runBuffered uses,
      // applied to THIS member's provider. A chain may mix members that
      // need emulation (non-Anthropic upstream) with ones that don't
      // (api.anthropic.com); each gets the right treatment in turn.
      const wsCap = resolved.capabilities.web_search
      const memberNeedsSearch =
        clientApi === 'anthropic-messages' &&
        !isAnthropicNative(member.provider) &&
        hasAnthropicWebSearchTool(workingReq) &&
        wsCap?.via === 'local' &&
        typeof wsCap.providerId === 'string' &&
        wsCap.providerId !== ''

      let memberOk: boolean
      if (memberNeedsSearch) {
        const rewritten = rewriteAnthropicWebSearchTool(workingReq)
        const providerIdOverride =
          wsCap?.via === 'local' && wsCap.providerId ? wsCap.providerId : undefined
        const outcome = await runWebSearchEmulationLoop({
          member,
          clientApi,
          clientRequest: rewritten,
          ctx: execCtx,
          ...(providerIdOverride ? { providerIdOverride } : {}),
        })
        for (const a of outcome.attempts) attempts.push({ ...a, step: step++ })
        memberOk = outcome.ok
        const last = outcome.attempts[outcome.attempts.length - 1]
        if (outcome.ok && last?.result.ok) {
          logger.info(
            `routed ${ctx.routeKey} → ${member.model.id} (${member.provider.id}) ` +
              `[web_search emulation, ${outcome.searches.size} search(es)]`,
          )
          winner = { ir: last.result.ir, status: outcome.status }
        } else {
          logger.warn(`routing: ${ctx.routeKey} → ${member.model.id} web_search emulation failed`)
        }
      } else {
        const result = await execAttempt(member, clientApi, workingReq, execCtx)
        attempts.push({ member, result, role, step: step++ })
        memberOk = result.ok
        if (result.ok) {
          logger.info(`routed ${ctx.routeKey} → ${member.model.id} (${member.provider.id})`)
          winner = { ir: result.ir, status: result.status }
        } else {
          logger.warn(
            `routing: ${ctx.routeKey} → ${member.model.id} failed ` +
              `(${result.status}): ${result.errorText}`,
          )
        }
      }

      if (memberOk) break
      // Advance to the next member only on an explicit error switch rule.
      if (isLast || !member.switchOn.some((r) => r.kind === 'error')) break
    }

    const response = buildChainResponse(clientApi, winner, attempts, wantsStream)

    // Capture after the response is built so the client is never blocked on it.
    await captureChain(
      {
        agent: ctx.agent,
        decoder: ctx.decoder,
        ...(ctx.instanceId ? { instanceId: ctx.instanceId } : {}),
      },
      clientApi,
      req,
      attempts,
      winner,
      resolved.configuredTarget,
      { ts: orchStartWall, durMs: performance.now() - orchT0 },
      risk,
    )

    return response
  } finally {
    endRequest()
  }
}

/** Render a chain walk's outcome as a client response. A `stream:true`
 *  client gets a synthesized SSE body — the buffered walk cannot true-stream. */
function buildChainResponse(
  clientApi: ModelApi,
  winner: { ir: IRResponse; status: number } | undefined,
  attempts: RoutedAttempt[],
  wantsStream: boolean,
): Response {
  if (winner) {
    if (wantsStream) {
      return new Response(synthesizeSse(clientApi, winner.ir), {
        status: winner.status,
        headers: { 'content-type': 'text/event-stream' },
      })
    }
    return new Response(JSON.stringify(serializeResponseFromIR(clientApi, winner.ir)), {
      status: winner.status,
      headers: { 'content-type': 'application/json' },
    })
  }

  const last = attempts[attempts.length - 1]
  const status = last && !last.result.ok && last.result.status >= 400 ? last.result.status : 502
  const detail = last && !last.result.ok ? last.result.errorText : 'no upstream attempt completed'
  return new Response(JSON.stringify({ error: 'agentfw: routed upstream call failed', detail }), {
    status,
    headers: { 'content-type': 'application/json' },
  })
}

// ── fusion path: parallel panel → judge → synthesis ──────────────
//
// agentfw's local take on OpenRouter Fusion. The combo's panel answers the
// prompt in parallel — each slot is its own failover chain (a token/USD cap or
// an upstream error switches the slot to its fallback model). One combo-level
// vision companion bridges images for any text-only panel member (the part
// OpenRouter doesn't do); a judge distils the answers into a structured
// analysis; a synthesizer writes the final answer grounded in it. The client
// sees one model call; the whole fan-out is captured as a parent ($0, model =
// combo id) plus one child per upstream call (vision / panel / judge /
// synthesis), so cost rolls up as the sum.
//
// Inherent to fusion: nothing streams to the client until the panel and judge
// have finished, so a `stream:true` client gets a synthesized SSE of the final
// answer (the same synth-SSE the chain path uses). Latency = slowest panel
// member + judge + synthesis; cost = the sum of every call.

const FUSION_JUDGE_MAX_TOKENS = 2048

const FUSION_JUDGE_SYSTEM = [
  'You are the judge in a multi-model deliberation.',
  'You will be given the original request and several independent model answers to it.',
  'Compare the answers and respond with ONLY a JSON object (no prose, no markdown fence)',
  'of the shape: {"consensus":[string],"contradictions":[{"topic":string,"stances":[string]}],',
  '"partial_coverage":[{"models":[string],"point":string}],',
  '"unique_insights":[{"model":string,"insight":string}],"blind_spots":[string]}.',
  'Treat points all or most models agree on as higher-confidence consensus, surface genuine',
  'contradictions, preserve unique insights from individual models, and note blind spots',
  'no answer addressed.',
].join(' ')

const FUSION_SYNTH_GUIDANCE = [
  'Several models independently answered your task; their answers',
  '(and, where available, a judge’s structured analysis of them) follow.',
  'Synthesize the single best, complete final answer, leaning on the consensus,',
  'incorporating the strongest unique insights, and resolving any contradictions with',
  'your own judgement. Do not mention this deliberation, the panel, or the judge —',
  'answer the user directly.',
].join(' ')

type PanelAnswer = { model: string; text: string }

type PanelOutcome = {
  attempts: RoutedAttempt[]
  answer?: PanelAnswer
  winner?: { ir: IRResponse; status: number }
}

/** Concatenate an IR response's text blocks. */
function textOfIR(ir: IRResponse): string {
  return ir.blocks
    .filter((b): b is { type: 'text'; text: string } => b.type === 'text')
    .map((b) => b.text)
    .join('\n')
    .trim()
}

/** The last user turn's text, used to ground the judge prompt. */
function lastUserText(clientApi: ModelApi, req: Record<string, unknown>): string {
  try {
    const ir = parseRequestToIR(clientApi, req)
    for (let i = ir.messages.length - 1; i >= 0; i--) {
      const m = ir.messages[i]
      if (m?.role !== 'user') continue
      const t = m.content
        .filter((b): b is { type: 'text'; text: string } => b.type === 'text')
        .map((b) => b.text)
        .join('\n')
        .trim()
      if (t) return t
    }
  } catch {
    // unparseable — judge still works from the panel answers alone
  }
  return ''
}

/** Render the panel answers as a labelled block for the judge / synthesizer. */
function renderPanelAnswers(answers: PanelAnswer[]): string {
  return answers.map((a, i) => `[Model ${i + 1}: ${a.model}]\n${a.text}`).join('\n\n')
}

/** Run one panel slot — its own failover chain. The primary's `switchOn` rules
 *  (token/USD cap, error) hand off to the fallback. A text-only member gets the
 *  pre-described request (when the fusion's vision companion produced one);
 *  multimodal members keep the raw images. Per-member web_search emulation uses
 *  the fusion's pinned provider. Never throws — a fully-failed slot contributes
 *  no answer. */
async function runPanelMember(
  pm: ResolvedPanelMember,
  clientApi: ModelApi,
  req: Record<string, unknown>,
  describedReq: Record<string, unknown> | undefined,
  webSearchProviderId: string | undefined,
  execCtx: { agent: AgentId; reqHeaders: Headers },
): Promise<PanelOutcome> {
  const { members } = pm
  const attempts: RoutedAttempt[] = []
  let answer: PanelAnswer | undefined
  let winner: { ir: IRResponse; status: number } | undefined

  for (let i = 0; i < members.length; i++) {
    const member = members[i]!
    const isLast = i === members.length - 1

    // Quota pre-checks on the member we might switch FROM — skip it (fail over to
    // the fallback) when it's over its daily/monthly USD or token cap.
    if (!isLast) {
      const budgetRule = member.switchOn.find(
        (r): r is Extract<SwitchRule, { kind: 'budget' }> => r.kind === 'budget',
      )
      if (budgetRule && (await spendInPeriod(member.model.id, budgetRule.period)) >= budgetRule.usdLimit) {
        continue
      }
      const tokenRule = member.switchOn.find(
        (r): r is Extract<SwitchRule, { kind: 'tokens' }> => r.kind === 'tokens',
      )
      if (tokenRule && (await tokensInPeriod(member.model.id, tokenRule.period)) >= tokenRule.tokenLimit) {
        continue
      }
    }

    const memberTextOnly = !(member.model.input ?? []).includes('image')
    const memberReq = memberTextOnly && describedReq ? describedReq : req

    const needsSearch =
      webSearchProviderId != null &&
      clientApi === 'anthropic-messages' &&
      !isAnthropicNative(member.provider) &&
      hasAnthropicWebSearchTool(memberReq)

    let ok: boolean
    if (needsSearch) {
      const rewritten = rewriteAnthropicWebSearchTool(memberReq)
      const outcome = await runWebSearchEmulationLoop({
        member,
        clientApi,
        clientRequest: rewritten,
        ctx: execCtx,
        providerIdOverride: webSearchProviderId,
      })
      for (const a of outcome.attempts) attempts.push({ ...a, role: 'panel' })
      const last = outcome.attempts[outcome.attempts.length - 1]
      ok = outcome.ok && !!last?.result.ok
      if (ok && last?.result.ok) {
        winner = { ir: last.result.ir, status: outcome.status }
        answer = { model: member.model.id, text: textOfIR(last.result.ir) }
      }
    } else {
      const result = await execAttempt(member, clientApi, memberReq, execCtx)
      attempts.push({ member, result, role: 'panel', step: 0, request: { api: clientApi, body: memberReq } })
      ok = result.ok
      if (result.ok) {
        winner = { ir: result.ir, status: result.status }
        answer = { model: member.model.id, text: textOfIR(result.ir) }
      }
    }

    if (ok) break
    // Advance to the fallback only on an explicit error switch rule.
    if (isLast || !member.switchOn.some((r) => r.kind === 'error')) break
  }

  return { attempts, ...(answer ? { answer } : {}), ...(winner ? { winner } : {}) }
}

/** Run the judge: distil the panel answers into a structured analysis. The
 *  attempt is captured regardless; on failure the analysis is absent and the
 *  synthesizer works from the raw answers (matches OpenRouter's judge-failure
 *  behaviour). */
async function runJudge(
  judge: ModelRef,
  clientApi: ModelApi,
  req: Record<string, unknown>,
  answers: PanelAnswer[],
  execCtx: { agent: AgentId; reqHeaders: Headers },
): Promise<{ attempt: RoutedAttempt; analysis?: string }> {
  const member: ResolvedMember = {
    model: judge.model,
    provider: judge.provider,
    api: judge.api,
    switchOn: [],
  }
  const userText = lastUserText(clientApi, req)
  const judgeIR: IRRequest = {
    model: judge.model.id,
    system: FUSION_JUDGE_SYSTEM,
    messages: [
      {
        role: 'user',
        content: [
          {
            type: 'text',
            text: `Original request:\n${userText}\n\nPanel answers:\n${renderPanelAnswers(answers)}`,
          },
        ],
      },
    ],
    maxTokens: FUSION_JUDGE_MAX_TOKENS,
    stream: false,
  }
  const body = serializeRequestFromIR(judge.api, judgeIR) as Record<string, unknown>
  const result = await execAttempt(member, judge.api, body, execCtx)
  const attempt: RoutedAttempt = {
    member,
    result,
    role: 'judge',
    step: 0,
    request: { api: judge.api, body },
  }
  return result.ok ? { attempt, analysis: textOfIR(result.ir) } : { attempt }
}

/** Run the synthesizer: the agent's original request plus a final user turn
 *  carrying the panel answers and the judge analysis, instructing the model to
 *  write the grounded final answer. A text-only synthesizer builds on the
 *  pre-described request (images already turned to text) so it never receives
 *  an image it can't handle. Falls back to the raw request if the IR round-trip
 *  fails. */
async function runSynthesis(
  synth: ModelRef,
  clientApi: ModelApi,
  req: Record<string, unknown>,
  describedReq: Record<string, unknown> | undefined,
  answers: PanelAnswer[],
  analysis: string | undefined,
  execCtx: { agent: AgentId; reqHeaders: Headers },
): Promise<{ attempt: RoutedAttempt; winner?: { ir: IRResponse; status: number } }> {
  const member: ResolvedMember = {
    model: synth.model,
    provider: synth.provider,
    api: synth.api,
    switchOn: [],
  }
  // A text-only synthesizer can't take the raw image — use the pre-described
  // request when one exists. A multimodal synthesizer keeps the real image.
  const synthTextOnly = !(synth.model.input ?? []).includes('image')
  const base = synthTextOnly && describedReq ? describedReq : req
  let body: Record<string, unknown>
  try {
    const ir = parseRequestToIR(clientApi, base)
    const deliberation =
      `${FUSION_SYNTH_GUIDANCE}\n\nPanel answers:\n${renderPanelAnswers(answers)}` +
      (analysis ? `\n\nJudge analysis:\n${analysis}` : '')
    const messages: IRMessage[] = mergeConsecutive([
      ...ir.messages,
      { role: 'user', content: [{ type: 'text', text: deliberation }] },
    ])
    const synthIR: IRRequest = { ...ir, model: synth.model.id, messages, stream: false }
    body = serializeRequestFromIR(clientApi, synthIR) as Record<string, unknown>
  } catch {
    body = base
  }
  const result = await execAttempt(member, clientApi, body, execCtx)
  const attempt: RoutedAttempt = {
    member,
    result,
    role: 'synthesis',
    step: 0,
    request: { api: clientApi, body },
  }
  return result.ok ? { attempt, winner: { ir: result.ir, status: result.status } } : { attempt }
}

/** Drive a fusion combo: fan out the panel, judge their answers, synthesize the
 *  final one. Buffered throughout (the synthesizer's answer is synth-SSE'd when
 *  the client asked to stream). Capture reuses the chain machinery — parent +
 *  one child per upstream call. */
async function runFusion(
  ctx: WireContext,
  resolved: Extract<ResolvedRoute, { kind: 'fusion' }>,
  req: Record<string, unknown>,
): Promise<Response> {
  const orchStartWall = Date.now()
  const orchT0 = performance.now()
  const wantsStream = req.stream === true
  const { clientApi, panel, judge, synthesizer } = resolved
  const execCtx = { agent: ctx.agent, reqHeaders: ctx.reqHeaders }

  // Misconfiguration guard: the request carries an image, but no panel model can
  // see images and no multimodal companion is configured. Don't silently hand
  // an image to a text-only model (or auto-pick some unrelated vision model) —
  // fail with an actionable message telling the user to configure one.
  if (requestHasImageBlock(clientApi, req) && !resolved.vision) {
    const panelCanSeeImages = panel.some((slot) =>
      (slot.members[0]?.model.input ?? []).includes('image'),
    )
    if (!panelCanSeeImages) {
      const names = panel.map((slot) => slot.members[0]?.model.id).filter(Boolean).join(', ')
      logger.warn(
        `routing: ${ctx.routeKey} fusion "${resolved.configuredTarget.id}" received an image but ` +
          `no panel model can see images and no multimodal model is configured (${names})`,
      )
      return new Response(
        JSON.stringify({
          error:
            `agentfw: model fusion "${resolved.configuredTarget.id}" received an image, but none of ` +
            `its panel models (${names}) can see images and no multimodal model is configured. ` +
            `Set a Multimodal model on this fusion (Model Fusion tab) so agentfw can describe ` +
            `images for the text-only models, or add a vision-capable model to the panel.`,
        }),
        { status: 400, headers: { 'content-type': 'application/json' } },
      )
    }
  }

  beginRequest()
  try {
    const attempts: RoutedAttempt[] = []

    // Vision bridge — described once for the whole fusion (the combo's single
    // companion), reused by every text-only consumer (panel members AND the
    // synthesizer, which builds on the full request). Multimodal consumers keep
    // the raw images. Skipped when nothing text-only would receive an image.
    let describedReq: Record<string, unknown> | undefined
    if (resolved.vision && requestHasImageBlock(clientApi, req)) {
      const isTextOnly = (input: readonly string[] | undefined) => !(input ?? []).includes('image')
      const anyTextOnly =
        panel.some((slot) => slot.members.some((m) => isTextOnly(m.model.input))) ||
        isTextOnly(resolved.synthesizer.model.input)
      if (anyTextOnly) {
        const visionMember: ResolvedMember = {
          model: resolved.vision.model,
          provider: resolved.vision.provider,
          api: resolved.vision.api,
          switchOn: [],
        }
        const pre = await preDescribeImages(clientApi, req, visionMember, execCtx)
        for (const a of pre.attempts) attempts.push(a)
        describedReq = pre.request
      }
    }

    // Panel — every slot answers in parallel (each with its own failover).
    const outcomes = await Promise.all(
      panel.map((pm) =>
        runPanelMember(pm, clientApi, req, describedReq, resolved.webSearchProviderId, execCtx),
      ),
    )
    for (const o of outcomes) for (const a of o.attempts) attempts.push(a)
    const answers = outcomes
      .map((o) => o.answer)
      .filter((a): a is PanelAnswer => a != null && a.text !== '')

    let winner: { ir: IRResponse; status: number } | undefined
    if (answers.length === 0) {
      logger.warn(`routing: ${ctx.routeKey} fusion — every panel member failed`)
    } else {
      // Judge — best-effort; a failure just drops the analysis.
      const judged = await runJudge(judge, clientApi, req, answers, execCtx)
      attempts.push(judged.attempt)
      if (judged.analysis === undefined) {
        logger.warn(`routing: ${ctx.routeKey} fusion judge failed, synthesizing from raw answers`)
      }
      // Synthesis — the final answer. Fall back to the best panel answer if it fails.
      const synthed = await runSynthesis(
        synthesizer,
        clientApi,
        req,
        describedReq,
        answers,
        judged.analysis,
        execCtx,
      )
      attempts.push(synthed.attempt)
      winner = synthed.winner ?? outcomes.find((o) => o.winner)?.winner
      logger.info(
        `routed ${ctx.routeKey} → fusion:${resolved.configuredTarget.id} ` +
          `[${answers.length}/${panel.length} panel, judge ${judged.analysis ? 'ok' : 'failed'}, ` +
          `synth ${synthed.winner ? 'ok' : 'failed'}]`,
      )
    }

    const renumbered = attempts.map((a, i) => ({ ...a, step: i }))
    const response = buildChainResponse(clientApi, winner, renumbered, wantsStream)
    await captureChain(
      {
        agent: ctx.agent,
        decoder: ctx.decoder,
        ...(ctx.instanceId ? { instanceId: ctx.instanceId } : {}),
      },
      clientApi,
      req,
      renumbered,
      winner,
      resolved.configuredTarget,
      { ts: orchStartWall, durMs: performance.now() - orchT0 },
      [],
    )
    return response
  } finally {
    endRequest()
  }
}

// ── streaming path: single-model cross-protocol swap ──────────────

/** True-stream a single-model cross-protocol swap: translate the upstream SSE
 *  to the client's wire format live, and tee a copy off the hot path to parse
 *  into IR for capture. A failed attempt is captured and returned as an error
 *  JSON — streaming has no failover, so there is nothing to switch to. */
async function runStreamingSwap(
  ctx: WireContext,
  resolved: Extract<ResolvedRoute, { kind: 'model' }>,
  req: Record<string, unknown>,
): Promise<Response> {
  // Vision pre-describe runs in runBuffered (it makes its own upstream
  // calls before the routed model gets the text-only request), so a
  // streaming swap with images defers to the buffered path which then
  // synthesises SSE on the way out (same pattern the chain path uses for
  // wantsStream).
  if (
    resolved.capabilities.vision?.via === 'companion' &&
    requestHasImageBlock(resolved.clientApi, req)
  ) {
    return runBuffered(ctx, resolved, req)
  }

  // Web-search emulation is multi-turn — same constraint as
  // vision-loop. The user-visible failure mode this fixes: Claude
  // Desktop's inner WebSearch wrapper always sets stream:true, so
  // without this delegate the request would land here, get the
  // web_search_20250305 server tool stripped, and the model would
  // see no tool to call. Buffered + synth-SSE on the way out keeps
  // the client's stream:true happy.
  if (
    resolved.clientApi === 'anthropic-messages' &&
    !isAnthropicNative(resolved.provider) &&
    requestHasWebSearchServerTool(req)
  ) {
    return runBuffered(ctx, resolved, req)
  }

  const orchStartWall = Date.now()
  const orchT0 = performance.now()
  const clientApi = resolved.clientApi
  const configuredTarget = resolved.configuredTarget
  const member: ResolvedMember = {
    model: resolved.model,
    provider: resolved.provider,
    api: resolved.api,
    switchOn: [],
  }

  beginRequest()
  const attempt = await execStream(member, clientApi, req, {
    agent: ctx.agent,
    reqHeaders: ctx.reqHeaders,
  })

  if (!attempt.ok) {
    logger.warn(
      `routing: ${ctx.routeKey} → ${member.model.id} stream failed ` +
        `(${attempt.status}): ${attempt.errorText}`,
    )
    const result: AttemptResult = {
      ok: false,
      status: attempt.status,
      errorText: attempt.errorText,
      ...(attempt.errorHeaders ? { errorHeaders: attempt.errorHeaders } : {}),
      durMs: attempt.durMs,
      startedAtWall: attempt.startedAtWall,
      upstreamUrl: attempt.upstreamUrl,
    }
    await captureSingle(
      {
        agent: ctx.agent,
        decoder: ctx.decoder,
        ...(ctx.instanceId ? { instanceId: ctx.instanceId } : {}),
      },
      clientApi,
      req,
      { member, result, role: 'primary', step: 0 },
      configuredTarget,
    )
    endRequest()
    const status = attempt.status >= 400 ? attempt.status : 502
    return new Response(
      JSON.stringify({
        error: 'agentfw: routed upstream call failed',
        detail: attempt.errorText,
      }),
      { status, headers: { 'content-type': 'application/json' } },
    )
  }

  logger.info(`routed ${ctx.routeKey} → ${member.model.id} (${member.provider.id}) [stream]`)

  const [toClient, toCapture] = attempt.body.tee()

  // Capture off the hot path — parse the teed copy into IR and record it.
  void (async () => {
    try {
      const ir = await parseStreamToIR(member.api, toCapture)
      const result: AttemptResult = {
        ok: true,
        status: attempt.status,
        json: null,
        ir,
        durMs: performance.now() - orchT0,
        startedAtWall: orchStartWall,
        upstreamUrl: attempt.upstreamUrl,
      }
      await captureSingle(
        {
          agent: ctx.agent,
          decoder: ctx.decoder,
          ...(ctx.instanceId ? { instanceId: ctx.instanceId } : {}),
        },
        clientApi,
        req,
        { member, result, role: 'primary', step: 0 },
        configuredTarget,
      )
    } catch (err) {
      logger.error(`orchestrator: stream capture failed: ${(err as Error).message}`)
    }
  })()

  const translated = translateSseStream(member.api, clientApi, toClient)
  return new Response(trackStream(translated, endRequest), {
    status: attempt.status,
    headers: { 'content-type': 'text/event-stream' },
  })
}

// ── fast path: same-protocol single-model swap ────────────────────

async function runSameProtocolSwap(
  ctx: WireContext,
  resolved: Extract<ResolvedRoute, { kind: 'model' }>,
  reqBody: ArrayBuffer,
): Promise<Response> {
  const { model, provider } = resolved
  const upstreamUrl = generationUrl(provider.baseUrl, resolved.api)
  // Fast-path body prep: rewrite the model id, then drop Anthropic
  // server tools when the destination isn't api.anthropic.com — same
  // reasoning as buildUpstreamRequest, just on raw bytes since this
  // path skips translation. Both helpers are no-ops when there's
  // nothing to change.
  let body = rewriteModel(reqBody, model.id)
  if (!isAnthropicNative(provider)) {
    body = stripAnthropicServerToolsFromBody(body)
  }
  // Same as buildUpstreamRequest's clamp, on raw bytes: keep the output
  // budget inside the routed model's declared context window.
  body = clampOutputBudgetBytes(body, resolved.api, model)

  // Credential masking, scoped to this provider — swap real secrets for fakes
  // before the body leaves the machine; restore them on the client branch
  // below (the decoder branch keeps the masked bytes, like passthrough).
  const masked = maskRequestBody(body, provider.id)
  if (masked) body = masked.body

  const headers = filterRequestHeaders(ctx.reqHeaders, new URL(upstreamUrl).host)
  dropSwapIncompatibleBetas(headers)
  if (provider.auth.kind === 'passthrough') {
    const extra = await dynamicHeadersFor(ctx.agent)
    for (const [k, v] of Object.entries(extra)) headers.set(k, v)
  } else {
    await applyAuth(headers, provider.auth, provider.id)
  }

  const t0 = performance.now()
  beginRequest()
  let upstreamRes: Response
  try {
    upstreamRes = await fetch(new Request(upstreamUrl, { method: 'POST', headers, body }))
  } catch (err) {
    endRequest()
    logger.error(
      `orchestrator: upstream fetch failed for ${ctx.routeKey} → ${model.id}: ${(err as Error).message}`,
    )
    return new Response(
      JSON.stringify({
        error: 'agentfw: routed upstream fetch failed',
        detail: (err as Error).message,
      }),
      { status: 502, headers: { 'content-type': 'application/json' } },
    )
  }

  logger.info(
    resolved.downgradedFrom
      ? `routed ${ctx.routeKey} → ${model.id} (${provider.id}) [subagent ↓ from ${resolved.downgradedFrom}]`
      : `routed ${ctx.routeKey} → ${model.id} (${provider.id})`,
  )

  const orchestration: Orchestration = {
    role: 'primary',
    configuredTarget: resolved.configuredTarget,
  }

  const decoder = decoderFor(ctx.decoder)
  if (!decoder || !upstreamRes.body) {
    if (!upstreamRes.body) {
      endRequest()
      return new Response(null, {
        status: upstreamRes.status,
        headers: filterResponseHeaders(upstreamRes.headers),
      })
    }
    return new Response(trackStream(restoreResponseStream(upstreamRes.body, masked), endRequest), {
      status: upstreamRes.status,
      headers: filterResponseHeaders(upstreamRes.headers),
    })
  }

  // Same protocol in, same protocol out — tee straight to the client exactly
  // as passthrough does, and decode the other branch off the hot path.
  const [toClient, toDecoder] = upstreamRes.body.tee()
  const upstreamHeaders = upstreamRes.headers

  void decoder
    .decode({
      agent: ctx.agent,
      provider: ctx.provider,
      providerId: provider.id,
      upstreamUrl,
      reqMethod: 'POST',
      reqHeaders: ctx.reqHeaders,
      reqBody: body,
      ...(resolved.downgradedFrom ? { clientModel: resolved.downgradedFrom } : {}),
      ...(ctx.instanceId ? { instanceId: ctx.instanceId } : {}),
      resStatus: upstreamRes.status,
      resHeaders: upstreamHeaders,
      resBody: toDecoder,
      startedAt: t0,
      orchestration,
    })
    .catch((err) => {
      logger.error(`decoder ${ctx.decoder} failed: ${(err as Error).message}`)
    })

  return new Response(trackStream(restoreResponseStream(toClient, masked), endRequest), {
    status: upstreamRes.status,
    headers: filterResponseHeaders(upstreamHeaders),
  })
}
