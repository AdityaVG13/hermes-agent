import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

import { notifyError } from './notifications'
import {
  $activeSessionId,
  $currentFastMode,
  $currentReasoningEffort,
  $currentServiceTier,
  setCurrentFastMode,
  setCurrentReasoningEffort,
  setCurrentServiceTier
} from './session'
import { $sessionStates, sessionTileDelegate } from './session-states'

const STORAGE_KEY = 'hermes.desktop.model-presets'
const pendingWrites = new Map<string, Promise<unknown>>()
const confirmedValues = new Map<string, string>()

/** Remember options only within the route that gave them meaning. */
export interface ModelPreset {
  effort?: string
  fast?: boolean
  serviceTier?: string
}

export const modelPresetServiceTier = ({ fast, serviceTier }: ModelPreset): string | undefined =>
  serviceTier ?? (fast === undefined ? undefined : fast ? 'priority' : 'normal')

type RequestGateway = <T>(method: string, params?: Record<string, unknown>) => Promise<T>

export interface ModelPresetScope {
  connectionId: string
  profile: string
  /** Opaque owner-issued endpoint/protocol/credential-scope identity. Never a secret. */
  routeId: string
}

export const modelPresetKey = (provider: string, model: string, scope?: ModelPresetScope): string =>
  scope?.connectionId && scope.profile && scope.routeId
    ? JSON.stringify(['route-v2', scope.connectionId, scope.profile, scope.routeId, provider, model])
    : ''

function load(): Record<string, ModelPreset> {
  const raw = storedString(STORAGE_KEY)

  if (!raw) {
    return {}
  }

  try {
    const parsed = JSON.parse(raw)

    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? (parsed as Record<string, ModelPreset>) : {}
  } catch {
    return {}
  }
}

export const $modelPresets = atom<Record<string, ModelPreset>>(load())

/** A profile-home change invalidates the owner's opaque route tokens. */
export function dropModelPresetsForProfile(profile: string, connectionId = 'local'): void {
  const entries = Object.entries($modelPresets.get())

  const kept = entries.filter(([key]) => {
    try {
      const binding = JSON.parse(key)

      return (
        !Array.isArray(binding) || binding[0] !== 'route-v2' || binding[1] !== connectionId || binding[2] !== profile
      )
    } catch {
      return true
    }
  })

  if (kept.length !== entries.length) {
    const next = Object.fromEntries(kept)
    $modelPresets.set(next)
    persistString(STORAGE_KEY, JSON.stringify(next))
  }
}

export function getModelPreset(provider: string, model: string, scope?: ModelPresetScope): ModelPreset {
  const key = modelPresetKey(provider, model, scope)

  return key ? ($modelPresets.get()[key] ?? {}) : {}
}

/** Merge a partial preset for one model and persist. */
export function setModelPreset(provider: string, model: string, patch: ModelPreset, scope?: ModelPresetScope): void {
  const key = modelPresetKey(provider, model, scope)

  // A legacy provider::model preference has no provable owner. Do not migrate or borrow it.
  if (!key) {
    return
  }

  const tier = modelPresetServiceTier(patch)

  const next = {
    ...$modelPresets.get(),
    [key]: {
      ...$modelPresets.get()[key],
      ...patch,
      ...(tier !== undefined ? { serviceTier: tier, fast: tier !== 'normal' } : {})
    }
  }

  $modelPresets.set(next)
  persistString(STORAGE_KEY, JSON.stringify(next))
}

/** Apply a model's preset to the composer, then push it to a live session.
 *  `undefined` skips that dimension; values are capability-gated upstream.
 *  Without a session the local draft still needs the preset, but must not call
 *  `config.set`: that falls back to persistent profile config when no session
 *  matches and would rewrite the user's defaults.
 *
 *  `primary: false` scopes the optimistic write to the tile's session slice —
 *  a tile's picker must not clobber the primary composer's effort/fast. */
export async function applyModelPreset(
  preset: ModelPreset,
  ctx: {
    failMessage: string
    primary?: boolean
    request: RequestGateway
    sessionId: null | string
    isCurrent?: (dimension: 'effort' | 'speed') => boolean
    scope?: string
    onFailure?: (dimension: 'effort' | 'speed', confirmed: ModelPreset) => void
  }
): Promise<void> {
  const { effort } = preset
  const tier = modelPresetServiceTier(preset)
  const fast = tier === undefined ? undefined : tier !== 'normal'
  const primary = ctx.primary ?? true
  const oldOwner = $activeSessionId.get()
  const slice = $sessionStates.get()[ctx.sessionId ?? '']

  const previous =
    primary && !slice
      ? { effort: $currentReasoningEffort.get(), fast: $currentFastMode.get(), serviceTier: $currentServiceTier.get() }
      : {
          effort: $sessionStates.get()[ctx.sessionId ?? '']?.reasoningEffort,
          fast: $sessionStates.get()[ctx.sessionId ?? '']?.fast,
          serviceTier: $sessionStates.get()[ctx.sessionId ?? '']?.serviceTier
        }

  const writeKey = (dimension: string) => `${ctx.scope ?? ''}::${ctx.sessionId}::${dimension}`

  if (primary) {
    if (effort !== undefined) {
      setCurrentReasoningEffort(effort)
    }

    if (fast !== undefined) {
      setCurrentFastMode(fast)
      setCurrentServiceTier(tier!)
    }
  }

  if (ctx.sessionId) {
    sessionTileDelegate()?.updateSession(ctx.sessionId, state => ({
      ...state,
      // Like setCurrentReasoningEffort: the wire level belongs to the previous
      // effort, so it stays unknown until the next session.info re-stamps it.
      ...(effort !== undefined ? { reasoningEffort: effort, reasoningEffortWire: '' } : {}),
      ...(fast !== undefined ? { fast, serviceTier: tier! } : {})
    }))
  }

  if (!ctx.sessionId) {
    return
  }

  // Each dimension has its own write/failure: an effort error must not skip
  // the speed request. A late failure cannot undo a newer edit or chat.
  await Promise.all(
    (
      [
        ['effort', effort],
        ['speed', tier]
      ] as const
    ).map(async ([dimension, value]) => {
      if (value === undefined || ctx.isCurrent?.(dimension) === false) {
        return
      }

      const key = writeKey(dimension)
      const preceding = pendingWrites.get(key)

      if (!preceding) {
        confirmedValues.set(
          key,
          dimension === 'effort'
            ? (previous.effort ?? '')
            : previous.serviceTier || (previous.fast ? 'priority' : 'normal')
        )
      }

      let write: Promise<void> | undefined

      try {
        write = Promise.resolve(preceding)
          .catch(() => {})
          .then(async () => {
            await ctx.request('config.set', {
              key: dimension === 'effort' ? 'reasoning' : 'fast',
              session_id: ctx.sessionId,
              value: dimension === 'speed' && value === 'priority' ? 'fast' : value
            })
            confirmedValues.set(key, value)
          })

        pendingWrites.set(key, write)

        await write
      } catch (err) {
        const confirmedValue = confirmedValues.get(writeKey(dimension)) ?? ''

        const confirmed: ModelPreset =
          dimension === 'effort'
            ? { effort: confirmedValue }
            : { serviceTier: confirmedValue || 'normal', fast: !!confirmedValue && confirmedValue !== 'normal' }

        ctx.onFailure?.(dimension, confirmed)

        if (ctx.isCurrent?.(dimension) ?? true) {
          if (primary && $activeSessionId.get() === oldOwner) {
            if (dimension === 'effort' && $currentReasoningEffort.get() === effort) {
              setCurrentReasoningEffort(confirmed.effort ?? '')
            } else if (dimension === 'speed' && $currentServiceTier.get() === tier) {
              setCurrentFastMode(confirmed.fast ?? false)
              setCurrentServiceTier(confirmed.serviceTier ?? '')
            }
          }

          if (ctx.sessionId && (!primary || $activeSessionId.get() === oldOwner)) {
            sessionTileDelegate()?.updateSession(ctx.sessionId, state => {
              if (dimension === 'effort' && state.reasoningEffort === effort) {
                return { ...state, reasoningEffort: confirmed.effort ?? '', reasoningEffortWire: '' }
              }

              if (dimension === 'speed' && state.serviceTier === tier) {
                return { ...state, fast: confirmed.fast ?? false, serviceTier: confirmed.serviceTier ?? '' }
              }

              return state
            })
          }
        }

        notifyError(err, ctx.failMessage)
      } finally {
        // Rollback must read the last confirmed value before the final writer
        // releases the chain. Draft/skipped writes never enter these maps.
        if (pendingWrites.get(key) === write) {
          pendingWrites.delete(key)
          confirmedValues.delete(key)
        }
      }
    })
  )
}
