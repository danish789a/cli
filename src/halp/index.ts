import { Command } from 'commander'

import { startAIHelp } from './command.js'

const program = new Command()

program
  .version('1.0.0')
  // .command('help:ai')
  .description('Start interactive AI help using documentation')
  .action(startAIHelp)

program.parse(process.argv)
