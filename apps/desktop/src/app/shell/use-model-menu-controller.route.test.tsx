import type { ModelOptionsResult } from '@hermes/shared'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, expect, test, vi } from 'vitest'

import { modelOptionsQueryKey } from '@/lib/model-options'
import { $modelPresets } from '@/store/model-presets'

import { useModelMenuController } from './use-model-menu-controller'

const catalog = (routeId: string): ModelOptionsResult => ({
  providers: [
    {
      slug: 'custom:lab',
      name: 'Lab',
      models: ['shared-model'],
      capabilities: { 'shared-model': { reasoning: true, fast: true, route_id: routeId } }
    }
  ]
})

afterEach(() => {
  cleanup()
  $modelPresets.set({})
})

test('the actual menu controller remembers settings only for the catalog owner and route binding', async () => {
  const client = new QueryClient({ defaultOptions: { queries: { staleTime: Infinity, retry: false } } })

  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )

  const requestGateway = vi.fn()

  const request = async (method: string, params?: Record<string, unknown>): Promise<never> => {
    requestGateway(method, params)
    throw new Error('Unexpected catalog request: this owner already has a fresh snapshot')
  }

  const first = { ownerConnectionId: 'local', profile: 'default' }

  const others = [
    { ownerConnectionId: 'remote', profile: 'default' },
    { ownerConnectionId: 'local', profile: 'worker' }
  ]

  for (const owner of [first, ...others]) {
    client.setQueryData(modelOptionsQueryKey(owner.profile, null, owner.ownerConnectionId), catalog('route-a'))
  }

  const { result, rerender, unmount } = renderHook(
    props => useModelMenuController({ ...props, onSelectModel: vi.fn(), requestGateway: request }),
    { initialProps: first, wrapper }
  )

  act(() =>
    result.current.controller.setOptions(
      { effort: 'max', fast: true },
      { isActive: false, provider: 'custom:lab', model: 'shared-model' }
    )
  )
  expect(result.current.controller.presetFor('custom:lab', 'shared-model')).toMatchObject({ effort: 'max', fast: true })

  for (const owner of others) {
    rerender(owner)
    expect(result.current.controller.presetFor('custom:lab', 'shared-model')).toEqual({})
  }

  rerender(first)
  expect(result.current.controller.presetFor('custom:lab', 'shared-model').effort).toBe('max')
  act(() =>
    client.setQueryData(modelOptionsQueryKey('default', null, 'local'), {
      providers: [{ ...catalog('route-a').providers[0], slug: 'lab', aliases: ['custom:lab', 'lab'] }]
    })
  )
  await waitFor(() => expect(result.current.controller.presetFor('lab', 'shared-model').effort).toBe('max'))
  act(() => client.setQueryData(modelOptionsQueryKey('default', null, 'local'), catalog('route-b')))
  await waitFor(() => expect(result.current.controller.presetFor('custom:lab', 'shared-model')).toEqual({}))
  expect(requestGateway).not.toHaveBeenCalled()
  unmount()
  client.clear()
})
