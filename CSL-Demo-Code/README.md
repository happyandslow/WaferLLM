# CSL-Demo-Code

Cerebras CSL (v1.4.0) Knowledge Base — documentation, tutorials, benchmarks, library source, and an MCP server for AI agent queries.

## Repository Structure

```
CSL-Demo-Code/
├── AI-Knowledge-Base/       # AI-curated topic summaries (query first)
├── Cerebras_Docs/           # Official documentation (markdown)
├── examples/
│   ├── tutorials/           # 33 tutorials (gemv, pipeline, topic, sdklayout)
│   ├── benchmarks/          # 22 benchmarks + shared libs
│   └── updates/             # Rolling additions from SDK updates
├── source-code/
│   └── csl_libraries/       # CSL standard library source (.csl)
├── csl-knowledge-server/    # MCP server (TypeScript)
└── CLAUDE.md                # Unified resource index for AI agents
```

## MCP Server Installation

The `csl-knowledge-server/` directory contains an MCP server that exposes the knowledge base to AI agents via the [Model Context Protocol](https://modelcontextprotocol.io/).

### Prerequisites

- Node.js >= 18
- npm

### Build

```bash
cd csl-knowledge-server
npm install
npm run build
```

### Register as Claude Code MCP Server

Add the server to your Claude Code configuration:

```bash
claude mcp add csl-knowledge -- node <absolute-path-to>/CSL-Demo-Code/csl-knowledge-server/dist/index.js
```

For example:

```bash
claude mcp add csl-knowledge -- node /home/user/CSL-Demo-Code/csl-knowledge-server/dist/index.js
```

You can verify it was added:

```bash
claude mcp list
```

### Available Tools

Once registered, the MCP server provides these tools to AI agents:

| Tool | Description |
|---|---|
| `search_knowledge` | Search AI-curated knowledge base (query first) |
| `search_docs` | Search official CSL documentation |
| `search_code` | Search example code and library source |
| `read_file` | Read any file from the knowledge base |
| `list_examples` | List tutorials and benchmarks |
| `lookup_feature` | Look up a CSL feature in the CLAUDE.md index |

### Quick Start (from scratch on a new server)

```bash
# 1. Clone the repository
git clone <repo-url> CSL-Demo-Code
cd CSL-Demo-Code

# 2. Build the MCP server
cd csl-knowledge-server
npm install
npm run build

# 3. Register with Claude Code
claude mcp add csl-knowledge -- node $(pwd)/dist/index.js
```
