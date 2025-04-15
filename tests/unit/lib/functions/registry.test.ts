import { mkdir, mkdtemp, writeFile } from 'fs/promises'
import { tmpdir } from 'os'
import { join } from 'path'

import { describe, expect, test, vi } from 'vitest'

import { FunctionsRegistry } from '../../../../dist/lib/functions/registry.js'
import { watchDebounced } from '../../../../dist/utils/command-helpers.js'
import { getFrameworksAPIPaths } from '../../../../dist/utils/frameworks-api.js'

const duplicateFunctions = [
  {
    filename: 'hello.js',
    content: `exports.handler = async (event) => ({ statusCode: 200, body: JSON.stringify({ message: 'Hello World from .js' }) })`,
  },
  {
    filename: 'hello.ts',
    content: `exports.handler = async (event) => ({ statusCode: 200, body: JSON.stringify({ message: 'Hello World from .ts' }) })`,
  },
  {
    filename: 'hello2.js',
    content: `exports.handler = async (event) => ({ statusCode: 200, body: JSON.stringify({ message: 'Hello World from .ts' }) })`,
  },
  {
    filename: 'hello2/main.go',
    subDir: 'hello2',
    content: `package main
    import (
      "fmt"
    )

    func main() {
      fmt.Println("Hello, world from a go function!")
    }
    `,
  },
]

vi.mock('../../../../dist/utils/command-helpers.js', async () => {
  const helpers = await vi.importActual('../../../../dist/utils/command-helpers.js')

  return {
    ...helpers,
    watchDebounced: vi.fn().mockImplementation(() => Promise.resolve({})),
  }
})

test('registry should only pass functions config to zip-it-and-ship-it', async () => {
  const projectRoot = '/projectRoot'
  const frameworksAPIPaths = getFrameworksAPIPaths(projectRoot)
  const functionsRegistry = new FunctionsRegistry({
    frameworksAPIPaths,
    projectRoot,
    // @ts-expect-error TS(2322) FIXME: Type 'string' is not assignable to type 'Plugin'.
    config: { functions: { '*': {} }, plugins: ['test'] },
  })
  // @ts-expect-error TS(2345) FIXME: Argument of type '() => void' is not assignable to... Remove this comment to see the full error message
  const prepareDirectoryScanStub = vi.spyOn(FunctionsRegistry, 'prepareDirectoryScan').mockImplementation(() => {})
  // @ts-expect-error TS(2345) FIXME: Argument of type '() => void' is not assignable to... Remove this comment to see the full error message
  const setupDirectoryWatcherStub = vi.spyOn(functionsRegistry, 'setupDirectoryWatcher').mockImplementation(() => {})
  // To verify that only the functions config is passed to zip-it-ship-it
  const listFunctionsStub = vi.spyOn(functionsRegistry, 'listFunctions').mockImplementation(() => Promise.resolve([]))

  // @ts-expect-error TS(2341) FIXME: Property 'projectRoot' is private and only accessi... Remove this comment to see the full error message
  await functionsRegistry.scan([functionsRegistry.projectRoot])

  expect(listFunctionsStub).toHaveBeenCalledOnce()
  expect(listFunctionsStub).toHaveBeenCalledWith(
    expect.anything(),
    // @ts-expect-error TS(2341) FIXME: Property 'config' is private and only accessible w... Remove this comment to see the full error message
    expect.objectContaining({ config: functionsRegistry.config.functions }),
  )

  await listFunctionsStub.mockRestore()
  await setupDirectoryWatcherStub.mockRestore()
  await prepareDirectoryScanStub.mockRestore()
})

describe('the registry handles duplicate functions based on extension precedence', () => {
  test('where .js takes precedence over .go, and .go over .ts', async () => {
    const projectRoot = await mkdtemp(join(tmpdir(), 'functions-extension-precedence'))
    const functionsDirectory = join(projectRoot, 'functions')
    await mkdir(functionsDirectory)

    duplicateFunctions.forEach(async (func) => {
      if (func.subDir) {
        const subDir = join(functionsDirectory, func.subDir)
        await mkdir(subDir)
      }
      const file = join(functionsDirectory, func.filename)
      await writeFile(file, func.content)
    })
    const functionsRegistry = new FunctionsRegistry({
      projectRoot,
      // @ts-expect-error TS(2322) FIXME: Type '{}' is not assignable to type 'NormalizedCac... Remove this comment to see the full error message
      config: {},
      timeouts: { syncFunctions: 1, backgroundFunctions: 1 },
      // @ts-expect-error TS(2322) FIXME: Type '{ port: number; }' is not assignable to type... Remove this comment to see the full error message
      settings: { port: 8888 },
      frameworksAPIPaths: getFrameworksAPIPaths(projectRoot),
    })
    // @ts-expect-error TS(2345) FIXME: Argument of type '() => void' is not assignable to... Remove this comment to see the full error message
    const prepareDirectoryScanStub = vi.spyOn(FunctionsRegistry, 'prepareDirectoryScan').mockImplementation(() => {})
    // @ts-expect-error TS(2345) FIXME: Argument of type '() => void' is not assignable to... Remove this comment to see the full error message
    const setupDirectoryWatcherStub = vi.spyOn(functionsRegistry, 'setupDirectoryWatcher').mockImplementation(() => {})

    await functionsRegistry.scan([functionsDirectory])
    // @ts-expect-error TS(2341) FIXME: Property 'functions' is private and only accessibl... Remove this comment to see the full error message
    const { functions } = functionsRegistry

    expect(functions.get('hello')).toHaveProperty('runtime.name', 'js')
    expect(functions.get('hello2')).toHaveProperty('runtime.name', 'go')

    await setupDirectoryWatcherStub.mockRestore()
    await prepareDirectoryScanStub.mockRestore()
  })
})

test('should add included_files to watcher', async () => {
  // @ts-expect-error TS(2345) FIXME: Argument of type '{ frameworksAPIPaths: Record<"co... Remove this comment to see the full error message
  const registry = new FunctionsRegistry({
    frameworksAPIPaths: getFrameworksAPIPaths('/project-root'),
  })
  const func = {
    name: '',
    config: { functions: { '*': { included_files: ['include/*', '!include/a.txt'] } } },
    build() {
      return { srcFilesDiff: { added: ['myfile'] }, includedFiles: ['include/*'] }
    },
    getRecommendedExtension() {},
    isTypeScript() {
      return false
    },
  }

  // @ts-expect-error TS(2345) FIXME: Argument of type '{ name: string; config: { functi... Remove this comment to see the full error message
  await registry.buildFunctionAndWatchFiles(func)

  expect(watchDebounced).toHaveBeenCalledOnce()
  expect(watchDebounced).toHaveBeenCalledWith(['myfile', 'include/*'], expect.anything())
})
