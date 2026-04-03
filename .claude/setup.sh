#!/bin/bash
# Ensure MCP server packages are globally installed
PACKAGES=(
  "@modelcontextprotocol/server-sequential-thinking"
  "@modelcontextprotocol/server-filesystem"
  "@modelcontextprotocol/server-memory"
  "@upstash/context7-mcp@latest"
  "@playwright/mcp@latest"
)
for pkg in "${PACKAGES[@]}"; do
  npm ls -g "$pkg" &>/dev/null || npm install -g "$pkg" &>/dev/null
done
