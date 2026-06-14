// Routing resolution: a routeKey "<agent>/<provider>" → what the orchestrator
// should actually do with the request. Reads the cached routing policy and
// model registry; never touches disk on the proxy path.
//
// Two resolved shapes drive the orchestrator's fast paths:
//   - `model`  → a 1-member chain. Same- or cross-protocol single-model swap.
//   - `chain`  → 2+ members. Failover walk with switch rules.
// A 0-member chain, an unknown model, or a non-translatable client decoder
// falls back to passthrough.

import { logger } from '../../core/logger.ts'
import {
  type CombinationModel,
  type ModelApi,
  type ModelEntry,
  type ProviderEntry,
  findCombo,
  findModel,
  findProvider,
  resolveApi,
} from '../../core/model-registry.ts'
import type { DecoderKind } from '../../core/routes.ts'
import {
  type CapabilityFulfillment,
  type CapabilityId,
  type ChainMember,
  type SwitchRule,
  routingFor,
} from '../../core/routing-policy.ts'
import { getSecret } from '../../core/secrets.ts'
import { getModelRegistry, getRoutingPolicy, getSecrets } from '../routing/load.ts'

/** One fully-resolved chain member: a model, its provider, its wire API, and
 *  the conditions that advance the chain past it. */
export type ResolvedMember = {
  model: ModelEntry
  provider: ProviderEntry
  api: ModelApi
  switchOn: SwitchRule[]
}

/** A capability fulfillment resolved against the registry. The
 *  orchestrator reads this map regardless of whether the route's main
 *  target is a single model or a chain — capability execution is
 *  orthogonal to model routing. */
export type ResolvedCapability =
  | { via: 'companion'; ref: ModelRef }
  | { via: 'local'; providerId?: string }

export type ResolvedCapabilities = Partial<Record<CapabilityId, ResolvedCapability>>

/** One resolved fusion panel slot: a failover chain. `members[0]` is the
 *  primary (carries the slot's switch rules); any following member is the
 *  fallback reached when a switch rule fires. */
export type ResolvedPanelMember = {
  members: ResolvedMember[]
}

export type ResolvedRoute =
  | { kind: 'passthrough' }
  | {
      kind: 'model'
      /** The wire format the client agent speaks. */
      clientApi: ModelApi
      model: ModelEntry
      provider: ProviderEntry
      /** The target model's wire format — may differ from clientApi. */
      api: ModelApi
      capabilities: ResolvedCapabilities
      configuredTarget: { kind: 'model'; id: string }
      /** Set when this route is a subagent cost-saver downgrade — the model
       *  the client originally requested (e.g. `claude-opus-4-8`), recorded so
       *  capture can show requested-vs-served and the dashboard the savings. */
      downgradedFrom?: string
    }
  | {
      kind: 'chain'
      clientApi: ModelApi
      members: ResolvedMember[]
      capabilities: ResolvedCapabilities
      /** `id` is the chain's entry-point model id — drives the parent
       *  packet's displayed label so the dashboard can show
       *  `chain: <first-member>` for the route decision. */
      configuredTarget: { kind: 'chain'; id: string }
    }
  | {
      kind: 'fusion'
      clientApi: ModelApi
      /** Panel slots, run in parallel; each slot is its own failover chain. */
      panel: ResolvedPanelMember[]
      /** One multimodal companion for the whole fusion — pre-describes images
       *  for text-only panel members. Absent → no bridge (panel is multimodal). */
      vision?: ModelRef
      /** web_search local provider pin for the panel (anthropic → non-anthropic). */
      webSearchProviderId?: string
      /** The judge that distils the panel answers into a structured analysis. */
      judge: ModelRef
      /** The model that writes the final answer grounded in the analysis. */
      synthesizer: ModelRef
      /** `id` is the combo id — drives the parent packet's label. */
      configuredTarget: { kind: 'combo'; id: string }
    }

/** A route's decoder → the wire API agentfw can translate, if any. */
export function decoderToApi(decoder: DecoderKind): ModelApi | undefined {
  if (decoder === 'anthropic') return 'anthropic-messages'
  if (decoder === 'openai-chat') return 'openai-chat'
  if (decoder === 'openai-responses') return 'openai-responses'
  return undefined
}

export type ModelRef = { model: ModelEntry; provider: ProviderEntry; api: ModelApi }

/** A seeded provider's managed auth points at a secret in secrets.json. If
 *  that secret is absent — never captured, or rotated out-of-band — fall
 *  back to passthrough for this request rather than injecting an empty
 *  credential, which would 401. The stale secret is re-captured on the
 *  next `agentfw wire`. */
function withResolvableAuth(provider: ProviderEntry): ProviderEntry {
  const auth = provider.auth
  // passthrough has nothing to resolve; agent-oauth resolves its token at
  // request time (with its own graceful fallback) — not from secrets.json.
  if (auth.kind === 'passthrough' || auth.kind === 'agent-oauth') return provider
  if (getSecret(getSecrets(), auth.valueRef) !== undefined) return provider
  logger.warn(
    `routing: secret "${auth.valueRef}" missing for provider "${provider.id}", using passthrough`,
  )
  return { ...provider, auth: { kind: 'passthrough' } }
}

/** Resolve a model id (optionally scoped to a provider) into its model,
 *  provider, and effective wire API. The providerId disambiguates when
 *  the same model id exists under multiple providers (e.g. Xiangxin-2XL
 *  harvested under hermes and also added by hand under a custom
 *  provider). Omitting it falls back to first-match-by-id so legacy
 *  routing-policy entries still resolve. */
export function resolveModelRef(modelId: string, providerId?: string): ModelRef | undefined {
  const reg = getModelRegistry()
  const model = findModel(reg, modelId, providerId)
  if (!model) return undefined
  const provider = findProvider(reg, model.providerId)
  if (!provider) return undefined
  const api = resolveApi(reg, model)
  if (!api) return undefined
  return { model, provider: withResolvableAuth(provider), api }
}

export function resolveRoute(routeKey: string, decoder: DecoderKind): ResolvedRoute {
  const routing = routingFor(getRoutingPolicy(), routeKey)
  const target = routing.target
  if (target.kind === 'passthrough') return { kind: 'passthrough' }

  const clientApi = decoderToApi(decoder)
  if (!clientApi) return { kind: 'passthrough' }

  // A `composite` target dereferences a registry combination model — now a
  // fusion panel + judge + synthesizer. The combo is the source of truth, so
  // the route's own `capabilities` are ignored.
  if (target.kind === 'composite') {
    const combo = findCombo(getModelRegistry(), target.comboId)
    if (!combo) {
      logger.warn(
        `routing: ${routeKey} → unknown combination model "${target.comboId}", passing through`,
      )
      return { kind: 'passthrough' }
    }
    return resolveFusion(routeKey, combo, clientApi)
  }

  // A `chain` target uses its inline members and the route's per-route
  // capabilities — the failover walk.
  const memberSpecs: ChainMember[] = target.members
  const capabilities = resolveCapabilityMap(routeKey, routing.capabilities)

  // resolve each member; drop unresolvable ones.
  const members: ResolvedMember[] = []
  for (const m of memberSpecs) {
    const ref = resolveModelRef(m.modelId, m.providerId)
    if (!ref) {
      const where = m.providerId ? `model "${m.providerId}/${m.modelId}"` : `model "${m.modelId}"`
      logger.warn(`routing: ${routeKey} chain member ${where} is unresolvable, skipping`)
      continue
    }
    members.push({ ...ref, switchOn: m.switchOn ?? [] })
  }
  if (members.length === 0) {
    logger.warn(`routing: ${routeKey} chain has no resolvable members, passing through`)
    return { kind: 'passthrough' }
  }

  // 1-member chain → the single-model fast path. switchOn on a lone member
  // can never fire (nothing to switch to), so it's safely dropped.
  if (members.length === 1) {
    const only = members[0]!
    return {
      kind: 'model',
      clientApi,
      model: only.model,
      provider: only.provider,
      api: only.api,
      capabilities,
      configuredTarget: { kind: 'model', id: only.model.id },
    }
  }

  return {
    kind: 'chain',
    clientApi,
    members,
    capabilities,
    configuredTarget: { kind: 'chain', id: members[0]!.model.id },
  }
}

/** Resolve a fusion combo into its panel, judge, and synthesizer. Unresolvable
 *  panel members are dropped (logged); an empty panel falls back to passthrough.
 *  The synthesizer defaults to the first panel member and the judge to the
 *  synthesizer, so a panel-only combo always resolves. An unresolvable
 *  configured judge/synthesizer falls back the same way rather than failing the
 *  whole route. */
function resolveFusion(
  routeKey: string,
  combo: CombinationModel,
  clientApi: ModelApi,
): ResolvedRoute {
  const panel: ResolvedPanelMember[] = []
  for (const m of combo.panel) {
    const primary = resolveModelRef(m.modelId, m.providerId)
    if (!primary) {
      const where = m.providerId ? `model "${m.providerId}/${m.modelId}"` : `model "${m.modelId}"`
      logger.warn(`routing: ${routeKey} fusion panel ${where} is unresolvable, skipping`)
      continue
    }
    const members: ResolvedMember[] = [{ ...primary, switchOn: m.switchOn ?? [] }]
    if (m.fallback) {
      const fb = resolveModelRef(m.fallback.modelId, m.fallback.providerId)
      if (fb) members.push({ ...fb, switchOn: [] })
      else
        logger.warn(
          `routing: ${routeKey} fusion fallback "${m.fallback.modelId}" unresolvable, ` +
            `${m.modelId} runs without failover`,
        )
    }
    panel.push({ members })
  }
  if (panel.length === 0) {
    logger.warn(`routing: ${routeKey} fusion combo has no resolvable panel members, passing through`)
    return { kind: 'passthrough' }
  }

  let vision: ModelRef | undefined
  if (combo.vision) {
    vision = resolveModelRef(combo.vision.modelId, combo.vision.providerId)
    if (!vision)
      logger.warn(
        `routing: ${routeKey} fusion vision companion "${combo.vision.modelId}" unresolvable, ` +
          'images will be dropped for text-only panel members',
      )
  }

  const firstPrimary = panel[0]!.members[0]!
  const firstRef: ModelRef = {
    model: firstPrimary.model,
    provider: firstPrimary.provider,
    api: firstPrimary.api,
  }

  let synthesizer = firstRef
  if (combo.synthesizer) {
    const ref = resolveModelRef(combo.synthesizer.modelId, combo.synthesizer.providerId)
    if (ref) synthesizer = ref
    else
      logger.warn(
        `routing: ${routeKey} fusion synthesizer "${combo.synthesizer.modelId}" unresolvable, ` +
          `falling back to ${firstPrimary.model.id}`,
      )
  }

  let judge = synthesizer
  if (combo.judge) {
    const ref = resolveModelRef(combo.judge.modelId, combo.judge.providerId)
    if (ref) judge = ref
    else
      logger.warn(
        `routing: ${routeKey} fusion judge "${combo.judge.modelId}" unresolvable, ` +
          `falling back to ${synthesizer.model.id}`,
      )
  }

  return {
    kind: 'fusion',
    clientApi,
    panel,
    ...(vision ? { vision } : {}),
    ...(combo.webSearch?.providerId ? { webSearchProviderId: combo.webSearch.providerId } : {}),
    judge,
    synthesizer,
    configuredTarget: { kind: 'combo', id: combo.id },
  }
}

function resolveCapabilityMap(
  routeKey: string,
  raw: Partial<Record<CapabilityId, CapabilityFulfillment>> | undefined,
): ResolvedCapabilities {
  if (!raw) return {}
  const out: ResolvedCapabilities = {}
  for (const [id, fulfillment] of Object.entries(raw)) {
    const resolved = resolveOneCapability(routeKey, id as CapabilityId, fulfillment)
    if (resolved) out[id as CapabilityId] = resolved
  }
  return out
}

function resolveOneCapability(
  routeKey: string,
  capabilityId: CapabilityId,
  f: CapabilityFulfillment,
): ResolvedCapability | undefined {
  if (f.via === 'local') {
    return { via: 'local', ...(f.providerId ? { providerId: f.providerId } : {}) }
  }
  const ref = resolveModelRef(f.modelId, f.providerId)
  if (!ref) {
    const where = f.providerId ? `model "${f.providerId}/${f.modelId}"` : `model "${f.modelId}"`
    logger.warn(
      `routing: ${routeKey} capability ${capabilityId} companion ${where} is unresolvable, ignoring`,
    )
    return undefined
  }
  return { via: 'companion', ref }
}
