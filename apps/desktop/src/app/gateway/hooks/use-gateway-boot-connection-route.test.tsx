import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { DesktopConnectionsRegistry } from '@/global'
import { $desktopBoot } from '@/store/boot'
import { $connectionsRegistry, _resetConnectionsForTests, selectConnection } from '@/store/connections'
import { closeSecondaryGateways, isActivePrimary } from '@/store/gateway'
import { endGatewaySwitch } from '@/store/gateway-switch'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $gatewayState } from '@/store/session'

import { FakeWebSocket } from '../../../test/fake-gateway-socket'

import { takeGatewaySurvivor } from './gateway-hmr-survivor'
import { useGatewayBoot } from './use-gateway-boot'

// Main resolves every profile-less getConnection() through the window route
// this hook records (electron/desktop-profile resolveDesktopConnectionRequest),
// so the route must follow the source the window actually shows.

vi.mock(import('@/store/notifications'), async importOriginal => ({
  ...(await importOriginal()),
  notifyError: vi.fn()
}))

vi.mock(import('@/store/terminal-backend-warning'), () => ({
  warnIfTerminalBackendUnavailable: vi.fn(async () => false)
}))

const primaryConn = {
  authMode: 'token' as const,
  baseUrl: 'https://vps.example.com',
  connectionId: 'primary-vps',
  profile: 'default',
  token: 't',
  wsUrl: 'wss://vps.example.com/api/ws?token=t'
}

const remoteConn = {
  authMode: 'token' as const,
  baseUrl: 'https://coder.example.com',
  connectionId: 'coder-remote',
  profile: 'default',
  token: 'c',
  wsUrl: 'wss://coder.example.com/api/ws?token=c'
}

const registry: DesktopConnectionsRegistry = {
  connections: [
    { id: 'primary-vps', kind: 'remote', label: 'VPS', tokenPreview: '...t', tokenSet: true },
    { id: 'coder-remote', kind: 'remote', label: 'Coder', tokenPreview: '...c', tokenSet: true }
  ],
  primary: 'primary-vps',
  secureTokenStorage: true,
  version: 2
}

function fakeDesktop() {
  const unsubscribe = () => () => undefined
  const descriptorFor = (connectionId: string) => (connectionId === remoteConn.connectionId ? remoteConn : primaryConn)

  return {
    api: vi.fn(async ({ path }: { path: string }) =>
      path === '/api/profiles/active' ? { active: 'default', current: 'default' } : { profiles: [] }
    ),
    connections: {
      list: vi.fn(async () => registry),
      setLastUsed: vi.fn(async (id: string) => ({ ok: true, registry: { ...registry, lastUsed: id } }))
    },
    getConnection: vi.fn(async () => primaryConn),
    getConnectionFor: vi.fn(async ({ connectionId, profile }: { connectionId: string; profile: string }) => ({
      ...descriptorFor(connectionId),
      connectionId,
      profile,
      registryScoped: true
    })),
    getGatewayWsUrl: vi.fn(async (conn?: { wsUrl?: string }) => conn?.wsUrl ?? primaryConn.wsUrl),
    getGatewayWsUrlFor: vi.fn(async ({ connectionId }: { connectionId: string }) => descriptorFor(connectionId).wsUrl),
    getBootProgress: vi.fn(async () => ({
      error: null,
      fakeMode: false,
      message: '',
      phase: 'init',
      progress: 0,
      retryable: false,
      running: true,
      timestamp: Date.now()
    })),
    onBootProgress: vi.fn(unsubscribe),
    onBackendExit: vi.fn(unsubscribe),
    onConnectionApplied: vi.fn(unsubscribe),
    onPowerResume: vi.fn(unsubscribe),
    onWindowStateChanged: vi.fn(unsubscribe),
    revalidateConnection: vi.fn(async () => ({ ok: true, rebuilt: false })),
    setActiveConnectionRoute: vi.fn(),
    touchBackend: vi.fn(async () => undefined),
    profile: { get: vi.fn(async () => ({ profile: 'default' })) }
  }
}

function Harness() {
  useGatewayBoot({
    beforeConnectionSwitch: () => undefined,
    handleGatewayEvent: () => undefined,
    handleServerRequest: () => false,
    onConnectionReady: () => undefined,
    onGatewayReady: () => undefined,
    refreshHermesConfig: async () => undefined,
    refreshSessions: async () => undefined
  })

  return null
}

function closeSurvivor() {
  try {
    takeGatewaySurvivor()?.gateway.close()
  } catch {
    // ignore
  }
}

const originalWebSocket = globalThis.WebSocket

beforeEach(() => {
  closeSurvivor()
  closeSecondaryGateways()
  $activeGatewayProfile.set('default')
  $connection.set(null)
  vi.useFakeTimers()
  FakeWebSocket.mode = 'open'
  FakeWebSocket.instances = []
  FakeWebSocket.pingMode = 'pong'
  ;(globalThis as { WebSocket: unknown }).WebSocket = FakeWebSocket
  $gatewayState.set('idle')
  $desktopBoot.set({
    error: null,
    fakeMode: false,
    message: '',
    phase: 'init',
    progress: 0,
    running: true,
    timestamp: Date.now(),
    visible: true
  })
})

afterEach(() => {
  cleanup()
  closeSurvivor()
  closeSecondaryGateways()
  _resetConnectionsForTests()
  $connectionsRegistry.set(null)
  $connection.set(null)
  endGatewaySwitch()
  vi.useRealTimers()
  ;(globalThis as { WebSocket: unknown }).WebSocket = originalWebSocket
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
})

async function flushAsync() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0)
  })
}

it("switching back to the primary's source moves main's window route back with $connection", async () => {
  // A return to the primary that published only $connection left main on the
  // remote, so the next primary reconnect dialed the remote under a UI that
  // showed the primary's source.
  const desktop = fakeDesktop()

  ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop
  render(<Harness />)
  await flushAsync()
  $connectionsRegistry.set(registry)

  for (const target of ['coder-remote', 'primary-vps']) {
    const switching = selectConnection(target)
    await flushAsync()
    await flushAsync()
    await switching
  }

  expect(isActivePrimary()).toBe(true)
  expect($connection.get()?.connectionId).toBe('primary-vps')
  expect(desktop.setActiveConnectionRoute).toHaveBeenCalledWith(
    expect.objectContaining({ connectionId: 'coder-remote' })
  )
  expect(desktop.setActiveConnectionRoute).toHaveBeenLastCalledWith(
    expect.objectContaining({ connectionId: 'primary-vps', profile: 'default' })
  )
})
