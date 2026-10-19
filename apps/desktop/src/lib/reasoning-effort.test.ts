import { DEFAULT_REASONING_EFFORT, REASONING_EFFORT_VALUES } from '@hermes/shared'
import { describe, expect, it } from 'vitest'

import {
  isThinkingEnabled,
  modelReasoningEfforts,
  reasoningEffortClamp,
  reasoningEffortLabel,
  resolveReasoningEffort
} from './reasoning-effort'

describe('reasoning-effort', () => {
  it('keeps Muse Code Contributor Max separate from the pay-per-token API limits', () => {
    expect(modelReasoningEfforts('muse-code', 'muse-spark-1.3-contributor')).toContain('max')
    expect(modelReasoningEfforts('meta-ai', 'muse-spark-1.3-contributor')).not.toContain('ultra')
    expect(modelReasoningEfforts('muse-code', 'muse-spark-1.3')).toContain('max')
    expect(modelReasoningEfforts('custom:relay', 'muse-spark-1.3-contributor')).toContain('max')
  })
  it('labels every level it claims to support', () => {
    for (const effort of REASONING_EFFORT_VALUES) {
      expect(reasoningEffortLabel(effort)).not.toBe('')
    }

    expect(reasoningEffortLabel('')).toBe('')
    // Unknown values pass through rather than silently reading as a real level.
    expect(reasoningEffortLabel('bogus')).toBe('bogus')
  })

  it('labels a route clamp from the gateway wire level only, never by inference', () => {
    expect(reasoningEffortLabel('ultra', 'max')).toBe('Ultra→Max')
    expect(reasoningEffortClamp('ultra', 'max')).toEqual({ effort: 'ultra', wire: 'max' })
    // Unknown ('' — not stamped yet / optimistic pick) or verbatim: plain label, no claim.
    expect(reasoningEffortLabel('ultra', '')).toBe('Ultra')
    expect(reasoningEffortLabel('ultra')).toBe('Ultra')
    expect(reasoningEffortLabel('high', 'high')).toBe('High')
    expect(reasoningEffortClamp('high', 'high')).toBeNull()
    expect(reasoningEffortClamp('none', '')).toBeNull()
  })

  it('treats empty as inherit and only `none` as off', () => {
    expect(isThinkingEnabled('none')).toBe(false)
    expect(isThinkingEnabled('high')).toBe(true)
    // Empty inherits the fallback, so an off fallback reads as off.
    expect(isThinkingEnabled('', 'none')).toBe(false)
    expect(isThinkingEnabled('', 'high')).toBe(true)
  })

  it('resolves a scale value: inherit, off, or clamp', () => {
    expect(resolveReasoningEffort('high')).toBe('high')
    // Empty inherits the profile default rather than snapping to medium.
    expect(resolveReasoningEffort('', 'ultra')).toBe('ultra')
    // Off selects nothing on the scale.
    expect(resolveReasoningEffort('none')).toBe('')
    expect(resolveReasoningEffort('bogus')).toBe(DEFAULT_REASONING_EFFORT)
  })
})
