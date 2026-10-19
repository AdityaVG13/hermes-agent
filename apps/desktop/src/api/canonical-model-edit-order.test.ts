// @vitest-environment node
import { expect, test } from 'vitest'

import { HermesGateway } from './client'

// Model row selection and preset/keyboard edits can overlap on the same socket.
// The authority accepts the first edit before its ACK is delivered; the next
// edit must wait for that ACK instead of sending the same CAS fence.
test('overlapping model and effort edits use successive owner fences', async () => {
  const wsPackage = 'ws'
  const { WebSocketServer } = await import(wsPackage)
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await new Promise<void>(resolve => server.once('listening', resolve))
  let revision = 1
  let generation = 2
  const mutations: any[] = []
  server.on('connection', (socket: any) => {
    socket.on('message', (bytes: Buffer) => {
      const frame = JSON.parse(bytes.toString())
      const p = frame.params
      const reply = (result: unknown) => socket.send(JSON.stringify({ jsonrpc: '2.0', id: frame.id, result }))

      if (frame.method === 'session.resume') {
        reply({ session_id: 's', revision, execution_generation: generation, messages: [], prompts: [] })
      } else if (frame.method === 'session.mutate') {
        mutations.push(p)

        if (p.expected_revision !== revision || p.expected_generation !== generation) {
          socket.send(JSON.stringify({ jsonrpc: '2.0', id: frame.id, error: {
            code: 4001, message: 'stale_generation', data: { reason: 'stale_generation' }
          } }))
        } else {
          revision += 1
          generation += 1
          const result = { session_id: 's', revision, execution_generation: generation, operation: p.operation, ...p.payload }
          setImmediate(() => reply(result))
        }
      }
    })
  })
  const client = new HermesGateway()

  try {
    const address = server.address() as { port: number }
    await client.connect(`ws://127.0.0.1:${address.port}/api/ws?native_dial=fixture&ticket=one-use`)
    await client.request('session.resume', { session_id: 's' })
    await Promise.all([
      client.request('config.set', { session_id: 's', key: 'model', value: 'grok-4.7 --provider xai-oauth' }),
      client.request('config.set', { session_id: 's', key: 'reasoning', value: 'xhigh' }),
      client.request('config.set', { session_id: 's', key: 'reasoning', value: 'high' })
    ])
    expect(mutations).toMatchObject([
      { operation: 'model', expected_revision: 1, expected_generation: 2 },
      { operation: 'reasoning', expected_revision: 2, expected_generation: 3 },
      { operation: 'reasoning', expected_revision: 3, expected_generation: 4 }
    ])
  } finally {
    client.close()

    for (const socket of server.clients) { socket.terminate() }
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})

test.each(['turn', 'policy-edit', 'repeated'] as const)('a %s generation race refreshes without overwriting policy or retrying forever', async race => {
  const wsPackage = 'ws'
  const { WebSocketServer } = await import(wsPackage)
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await new Promise<void>(resolve => server.once('listening', resolve))
  const edits: any[] = []
  let resumes = 0
  server.on('connection', (socket: any) => {
    socket.on('message', (bytes: Buffer) => {
      const frame = JSON.parse(bytes.toString())
      const reply = (result: unknown) => socket.send(JSON.stringify({ jsonrpc: '2.0', id: frame.id, result }))

      if (frame.method === 'session.resume') {
        resumes += 1
        reply({ session_id: 's', revision: race === 'policy-edit' && resumes > 1 ? 2 : 1,
          execution_generation: resumes, messages: [], prompts: [] })
      } else if (frame.method === 'session.mutate') {
        edits.push(frame.params)

        if (race === 'turn' && edits.length === 2) {
          reply({ session_id: 's', revision: 2, execution_generation: 3, effort: 'high' })
        } else {
          socket.send(JSON.stringify({ jsonrpc: '2.0', id: frame.id, error: {
            code: 4001, message: 'stale_generation', data: { reason: 'stale_generation' }
          } }))
        }
      }
    })
  })
  const client = new HermesGateway()

  try {
    const address = server.address() as { port: number }
    await client.connect(`ws://127.0.0.1:${address.port}/api/ws?native_dial=fixture&ticket=one-use`)
    await client.request('session.resume', { session_id: 's' })
    const edit = client.request('config.set', { session_id: 's', key: 'reasoning', value: 'high' })

    if (race === 'turn') { await expect(edit).resolves.toMatchObject({ value: 'high' }) }
    else { await expect(edit).rejects.toThrow('stale_generation') }

    expect(resumes).toBe(2)
    expect(edits).toHaveLength(race === 'policy-edit' ? 1 : 2)

    if (edits.length === 2) {
      expect(edits[1]).toMatchObject({ expected_generation: 2, expected_revision: 1 })
      expect(edits[1].request_id).not.toBe(edits[0].request_id)
    }
  } finally {
    client.close()

    for (const socket of server.clients) { socket.terminate() }
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})
