// Single source of truth for the installed agentfw version. Must match
// packages/agentfw/package.json. The CLI's --version flag and the update
// checker both read this — they must never disagree.
export const VERSION = '0.5.2'
