import { beforeEach, expect, test } from 'vitest'

import { $modelPresets, dropModelPresetsForProfile, getModelPreset, setModelPreset } from './model-presets'
import { dropTilesForProfile, migrateTilesForProfile } from './session-states'

const route = { connectionId: 'local', profile: 'default', routeId: 'endpoint-a:chat:credential-a' }

beforeEach(() => $modelPresets.set({}))

test('same-named models never restore another owner, endpoint, protocol or credential scope preset', () => {
  setModelPreset('custom:lab', 'shared-model', { effort: 'high', fast: true }, route)
  expect(getModelPreset('custom:lab', 'shared-model', route)).toMatchObject({ effort: 'high', fast: true })

  for (const other of [
    { ...route, connectionId: 'remote' },
    { ...route, profile: 'worker' },
    { ...route, routeId: 'endpoint-b:chat:credential-a' },
    { ...route, routeId: 'endpoint-a:responses:credential-a' },
    { ...route, routeId: 'endpoint-a:chat:credential-b' }
  ]) {
    expect(getModelPreset('custom:lab', 'shared-model', other)).toEqual({})
    setModelPreset('custom:lab', 'shared-model', { effort: 'low' }, other)
  }

  expect(getModelPreset('custom:lab', 'shared-model', route).effort).toBe('high')
})

test('legacy global presets and unknown routes cannot seed a scoped preference', () => {
  $modelPresets.set({ 'custom:lab::shared-model': { effort: 'max', fast: true } })
  expect(getModelPreset('custom:lab', 'shared-model', route)).toEqual({})
  expect(getModelPreset('custom:lab', 'shared-model')).toEqual({})
  setModelPreset('custom:lab', 'shared-model', { effort: 'low' })
  expect(getModelPreset('custom:lab', 'shared-model', route)).toEqual({})
})

test('profile cleanup removes only that connection and profile, never a namesake owner', () => {
  const remote = { ...route, connectionId: 'remote' }
  const worker = { ...route, profile: 'worker' }

  for (const scope of [route, remote, worker]) {
    setModelPreset('custom:lab', 'shared-model', { effort: 'high' }, scope)
  }

  dropModelPresetsForProfile('default', 'local')
  expect(getModelPreset('custom:lab', 'shared-model', route)).toEqual({})
  expect(getModelPreset('custom:lab', 'shared-model', remote).effort).toBe('high')
  expect(getModelPreset('custom:lab', 'shared-model', worker).effort).toBe('high')
})

test('profile deletion and rename retire only local bindings instead of reviving stale endpoint identities', () => {
  const worker = { ...route, profile: 'worker' }
  const remote = { ...worker, connectionId: 'remote' }
  setModelPreset('custom:lab', 'shared-model', { effort: 'max' }, worker)
  setModelPreset('custom:lab', 'shared-model', { effort: 'low' }, remote)
  migrateTilesForProfile('worker', 'renamed')
  expect(getModelPreset('custom:lab', 'shared-model', worker)).toEqual({})
  // The opaque token includes the old profile home; never transplant it to the new home.
  expect(getModelPreset('custom:lab', 'shared-model', { ...worker, profile: 'renamed' })).toEqual({})
  expect(getModelPreset('custom:lab', 'shared-model', remote).effort).toBe('low')
  setModelPreset('custom:lab', 'shared-model', { effort: 'high' }, worker)
  dropTilesForProfile('worker')
  expect(getModelPreset('custom:lab', 'shared-model', worker)).toEqual({})
  expect(getModelPreset('custom:lab', 'shared-model', remote).effort).toBe('low')
})
