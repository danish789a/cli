#!/usr/bin/env node
import { argv } from 'process'

import updateNotifier from 'update-notifier'

import { createMainCommand } from '../dist/commands/main.js'
import { logError } from '../dist/utils/command-helpers.js'
import getPackageJson from '../dist/utils/get-cli-package-json.js'
import { runProgram } from '../dist/utils/run-program.js'

// 12 hours
const UPDATE_CHECK_INTERVAL = 432e5
const pkg = await getPackageJson()

try {
  updateNotifier({
    pkg,
    updateCheckInterval: UPDATE_CHECK_INTERVAL,
  }).notify()
} catch (error) {
  logError(`Error checking for updates: ${error?.toString()}`)
}

const program = createMainCommand()

try {
  await runProgram(program, argv)

  program.onEnd()
} catch (error) {
  program.onEnd(error)
}
