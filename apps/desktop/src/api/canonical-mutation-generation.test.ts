import { expect, test } from 'vitest'

import { CanonicalDesktopProtocol } from './canonical-protocol'

const resume = (protocol: CanonicalDesktopProtocol, revision: number, generation: number, profile = 'default') =>
  protocol.result('session.resume', { session_id: 's', profile }, {
    session_id: 's', revision, execution_generation: generation
  })

test.each([
  { key: 'model', value: 'grok-4.7 --provider xai-oauth' },
  { key: 'reasoning', value: 'max' }
])('an acknowledged $key edit supplies the generation for the next edit without waiting for events', edit => {
  const protocol = new CanonicalDesktopProtocol()
  resume(protocol, 7, 3, 'work')
  const first = protocol.prepare('config.set', { session_id: 's', profile: 'work', ...edit })
  protocol.result('config.set', first, { session_id: 's', revision: 8, execution_generation: 4 })

  const next = protocol.prepare('config.set', { session_id: 's', profile: 'work', key: 'reasoning', value: 'high' })
  expect(next).toMatchObject({ expected_revision: 8, expected_generation: 4, profile: 'work' })

  // An exact retry's older receipt must not rewind either fence.
  protocol.result('config.set', first, { session_id: 's', revision: 7, execution_generation: 2 })
  expect(protocol.prepare('session.branch', { session_id: 's', profile: 'work' })).toMatchObject({
    expected_revision: 8, expected_generation: 4
  })
})

test('a known stale-generation refusal releases the exact request but a lost reply retains it', () => {
  const protocol = new CanonicalDesktopProtocol()
  resume(protocol, 7, 3)
  const edit = { session_id: 's', key: 'reasoning', value: 'max' }
  const first = protocol.prepare('config.set', edit)
  protocol.failure(first, new Error('connection closed'))
  expect(protocol.prepare('config.set', edit)).toEqual(first)

  protocol.failure(first, { data: { reason: 'stale_generation' } })
  resume(protocol, 7, 4)
  const retry = protocol.prepare('config.set', edit)
  expect(retry.request_id).not.toBe(first.request_id)
  expect(retry.expected_generation).toBe(4)
})

test('mutation generations remain scoped to their owning profile', () => {
  const protocol = new CanonicalDesktopProtocol()
  resume(protocol, 1, 2, 'a')
  resume(protocol, 5, 6, 'b')
  const edit = protocol.prepare('config.set', { session_id: 's', profile: 'a', key: 'reasoning', value: 'high' })
  protocol.result('config.set', edit, { session_id: 's', revision: 2, execution_generation: 3 })
  expect(protocol.prepare('session.branch', { session_id: 's', profile: 'b' })).toMatchObject({
    expected_revision: 5, expected_generation: 6
  })
})
