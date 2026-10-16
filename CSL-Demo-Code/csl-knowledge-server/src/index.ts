#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ListResourcesRequestSchema,
  ReadResourceRequestSchema,
  ErrorCode,
  McpError,
} from "@modelcontextprotocol/sdk/types.js";
import { execSync } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const KB_ROOT = process.argv[2] || path.resolve(import.meta.dirname, "..");

if (!fs.existsSync(KB_ROOT)) {
  console.error(`Knowledge base root not found: ${KB_ROOT}`);
  process.exit(1);
}

const DOCS_DIR = path.join(KB_ROOT, "Cerebras_Docs");
const EXAMPLES_DIR = path.join(KB_ROOT, "examples");
const SOURCE_DIR = path.join(KB_ROOT, "source-code");
const KNOWLEDGE_DIR = path.join(KB_ROOT, "AI-Knowledge-Base");
const CLAUDE_MD = path.join(KB_ROOT, "CLAUDE.md");

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Resolve a user-supplied relative path and ensure it stays inside KB_ROOT */
function resolvePath(relative: string): string {
  const resolved = path.resolve(KB_ROOT, relative);
  if (!resolved.startsWith(KB_ROOT + "/") && resolved !== KB_ROOT) {
    throw new McpError(ErrorCode.InvalidParams, "Path outside knowledge base");
  }
  return resolved;
}

/** Shell-escape a string for safe use in grep arguments */
function shellEscape(s: string): string {
  return "'" + s.replace(/'/g, "'\\''") + "'";
}

/** Run grep and return results. Returns empty string on no-match. */
function grepSearch(
  query: string,
  searchPath: string,
  opts: {
    includePattern?: string;
    maxResults?: number;
    contextLines?: number;
    ignoreCase?: boolean;
  } = {}
): string {
  const {
    includePattern,
    maxResults = 50,
    contextLines = 2,
    ignoreCase = false,
  } = opts;

  if (!fs.existsSync(searchPath)) return `Path not found: ${searchPath}`;

  const args: string[] = ["-rn", "--color=never"];
  if (ignoreCase) args.push("-i");
  if (includePattern) args.push(`--include=${shellEscape(includePattern)}`);
  if (contextLines > 0) args.push(`-C${contextLines}`);
  args.push("--", shellEscape(query), shellEscape(searchPath));

  try {
    const result = execSync(`grep ${args.join(" ")}`, {
      encoding: "utf-8",
      maxBuffer: 2 * 1024 * 1024,
      timeout: 15000,
    });

    const lines = result.split("\n");
    const limit = maxResults * (1 + 2 * contextLines);
    if (lines.length > limit) {
      return (
        lines.slice(0, limit).join("\n") +
        `\n\n... (truncated, showing first ${maxResults} matches out of many)`
      );
    }
    return result.trimEnd();
  } catch (e: unknown) {
    const err = e as { status?: number };
    if (err.status === 1) return "No matches found.";
    throw e;
  }
}

/** Get first non-empty, non-heading content line from a markdown file */
function getFirstSummaryLine(filePath: string): string {
  try {
    const content = fs.readFileSync(filePath, "utf-8");
    for (const line of content.split("\n")) {
      const t = line.trim();
      if (
        t &&
        !t.startsWith("#") &&
        !t.startsWith(">") &&
        !t.startsWith("---") &&
        !t.startsWith("```")
      ) {
        return t.length > 120 ? t.slice(0, 120) + "…" : t;
      }
    }
  } catch {
    /* ignore */
  }
  return "";
}

// ---------------------------------------------------------------------------
// Server-level instructions (D9)
// ---------------------------------------------------------------------------

const SERVER_INSTRUCTIONS = [
  "CSL Knowledge Server — Search Strategy Instructions",
  "",
  "**Recall-first principle**: Prefer returning extra content over missing relevant information.",
  "When uncertain, broaden the search scope and search multiple layers.",
  "",
  "**Layered search order** (follow this priority):",
  "1. search_knowledge — AI-curated summaries (query FIRST)",
  "2. search_docs — Official Cerebras documentation",
  "3. search_code — Tutorial + benchmark + library source code",
  "",
  "**WSE-2/WSE-3 awareness**: Always consider architecture differences.",
  "Use get_wse_differences when the user's question involves hardware-specific behavior.",
  "",
  "**File reading convention**: For tutorials/benchmarks, read in order: layout.csl → pe_program.csl → run.py",
  "",
  "**Do not stop at one layer** — if a search_knowledge result seems incomplete,",
  "continue to search_docs and search_code for comprehensive coverage.",
].join("\n");

// ---------------------------------------------------------------------------
// Server setup (D9: instructions, D10: resources capability)
// ---------------------------------------------------------------------------

const server = new Server(
  { name: "csl-knowledge-server", version: "2.0.0" },
  {
    capabilities: { tools: {}, resources: {} },
    instructions: SERVER_INSTRUCTIONS,
  }
);

// ---------------------------------------------------------------------------
// Resources (D10): Expose CLAUDE.md and _index.md
// ---------------------------------------------------------------------------

server.setRequestHandler(ListResourcesRequestSchema, async () => ({
  resources: [
    {
      uri: "csl-kb://CLAUDE.md",
      name: "CLAUDE.md — CSL Unified Resource Index",
      description:
        "Master index mapping CSL features to docs, examples, and library source. " +
        "Read this first to understand the knowledge base structure.",
      mimeType: "text/markdown",
    },
    {
      uri: "csl-kb://AI-Knowledge-Base/_index.md",
      name: "Knowledge Base Document Catalog",
      description:
        "Manifest of all AI-curated knowledge documents with summaries and keyword tags. " +
        "Use to quickly locate the right document for any CSL topic.",
      mimeType: "text/markdown",
    },
  ],
}));

server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  const { uri } = request.params;
  let filePath: string;

  if (uri === "csl-kb://CLAUDE.md") {
    filePath = CLAUDE_MD;
  } else if (uri === "csl-kb://AI-Knowledge-Base/_index.md") {
    filePath = path.join(KNOWLEDGE_DIR, "_index.md");
  } else {
    throw new McpError(ErrorCode.InvalidParams, `Unknown resource: ${uri}`);
  }

  if (!fs.existsSync(filePath)) {
    throw new McpError(
      ErrorCode.InternalError,
      `Resource file not found: ${filePath}`
    );
  }

  return {
    contents: [
      {
        uri,
        mimeType: "text/markdown",
        text: fs.readFileSync(filePath, "utf-8"),
      },
    ],
  };
});

// ---------------------------------------------------------------------------
// tools/list
// ---------------------------------------------------------------------------

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    // --- Existing tools with enhanced descriptions (D1, D7, D8) ---
    {
      name: "search_docs",
      description:
        "Search CSL official documentation (Cerebras_Docs/) by regex or keyword.\n\n" +
        "**When to use**: When you need official language specification details, API documentation, " +
        "or tutorial explanations that go beyond AI-curated summaries.\n\n" +
        "**Search rule**: Prefer recall over precision — better to return extra results than miss relevant content. " +
        "When uncertain, broaden the search scope.",
      inputSchema: {
        type: "object" as const,
        properties: {
          query: {
            type: "string",
            description: "Search query (supports grep regex)",
          },
          file_pattern: {
            type: "string",
            description: 'File glob filter (default "*.md")',
          },
          max_results: {
            type: "number",
            description: "Max matches to return (default 30)",
          },
          ignore_case: {
            type: "boolean",
            description: "Case-insensitive search (default true)",
          },
        },
        required: ["query"],
      },
    },
    {
      name: "search_code",
      description:
        "Search CSL example code (tutorials + benchmarks) and/or standard library source.\n\n" +
        "**When to use**: When you need concrete implementation patterns, usage examples of specific APIs, " +
        "or algorithm implementations.\n\n" +
        'Scope: "examples", "libraries", or "all".\n' +
        '**Tip**: Default searches .csl files. Use file_pattern "*.py" for host-side Python code (run.py). ' +
        'Use file_pattern "*" for all file types.\n\n' +
        "**Search rule**: Prefer recall over precision — better to return extra results than miss relevant content.",
      inputSchema: {
        type: "object" as const,
        properties: {
          query: {
            type: "string",
            description: "Search query (supports grep regex)",
          },
          scope: {
            type: "string",
            enum: ["examples", "libraries", "all"],
            description: 'Where to search (default "all")',
          },
          file_pattern: {
            type: "string",
            description:
              'File glob filter (default "*.csl"). Use "*.py" for host code, "*" for all.',
          },
          max_results: {
            type: "number",
            description: "Max matches to return (default 30)",
          },
        },
        required: ["query"],
      },
    },
    {
      name: "search_knowledge",
      description:
        "Search the AI-curated knowledge base (AI-Knowledge-Base/).\n\n" +
        "**When to use**: Use FIRST for any CSL topic. These summaries cover the essentials " +
        "(core concepts, code patterns, API quick-ref, best practices, common pitfalls). " +
        "Only search deeper layers if the summary is insufficient.\n\n" +
        "Returns available document listing alongside grep matches.\n\n" +
        "**Search rule**: Prefer recall over precision — better to return extra results than miss relevant content.",
      inputSchema: {
        type: "object" as const,
        properties: {
          query: {
            type: "string",
            description: "Search query (supports grep regex)",
          },
          max_results: {
            type: "number",
            description: "Max matches to return (default 30)",
          },
        },
        required: ["query"],
      },
    },
    {
      name: "read_file",
      description:
        "Read a file from the CSL knowledge base. Path is relative to the KB root.\n\n" +
        "**When to use**: Read a specific file in full after search results indicate it's relevant. " +
        "Also use to list directory contents when given a directory path.\n\n" +
        "**Search rule**: Prefer recall over precision — when in doubt, read the full file.",
      inputSchema: {
        type: "object" as const,
        properties: {
          path: {
            type: "string",
            description:
              'Relative path, e.g. "Cerebras_Docs/csl/language/dsds.md" or "examples/tutorials/gemv-06-routes-1/layout.csl"',
          },
          max_lines: {
            type: "number",
            description:
              "Max lines to return (default unlimited). Use for large files.",
          },
        },
        required: ["path"],
      },
    },
    {
      name: "list_examples",
      description:
        "List available CSL tutorials and/or benchmarks. Optionally filter by keyword.\n\n" +
        "**When to use**: Discover available tutorials and benchmarks by keyword. " +
        "Good for finding reference implementations.",
      inputSchema: {
        type: "object" as const,
        properties: {
          keyword: {
            type: "string",
            description:
              'Optional keyword filter (e.g. "gemv", "pipeline", "stencil")',
          },
          type: {
            type: "string",
            enum: ["tutorials", "benchmarks", "all"],
            description: 'What to list (default "all")',
          },
        },
      },
    },
    {
      name: "lookup_feature",
      description:
        "Look up a CSL language feature in the CLAUDE.md index.\n\n" +
        "**When to use**: Use as the STARTING POINT for any CSL feature lookup. " +
        "Returns an index of all relevant docs, examples, and source paths.\n\n" +
        "After receiving results, continue searching deeper layers for comprehensive coverage: " +
        "search_knowledge → search_docs → search_code.",
      inputSchema: {
        type: "object" as const,
        properties: {
          feature: {
            type: "string",
            description:
              'CSL feature keyword (e.g. "DSD", "task", "microthread", "memcpy", "color", "route", "FIFO", "switch")',
          },
        },
        required: ["feature"],
      },
    },
    // --- New tools (D4, D5, D6) ---
    {
      name: "get_topic_guide",
      description:
        "Get a complete layered search path for a CSL topic.\n\n" +
        "**When to use**: When you want a comprehensive, ordered list of ALL resources " +
        "for a topic across every knowledge layer. Returns: AI-Knowledge-Base doc → " +
        "official doc paths → tutorial paths → benchmark paths → library source paths.\n\n" +
        "Follow the returned path layer by layer to ensure nothing is missed.",
      inputSchema: {
        type: "object" as const,
        properties: {
          topic: {
            type: "string",
            description:
              'CSL topic keyword (e.g. "DSD", "routing", "FIFO", "memcpy", "microthread", "stencil", "FFT", "debugging")',
          },
        },
        required: ["topic"],
      },
    },
    {
      name: "search_all",
      description:
        "Search across ALL knowledge layers simultaneously.\n\n" +
        "**When to use**: When you don't know which layer contains the answer, or when " +
        "you want comprehensive results from every source. Searches: AI-Knowledge-Base + " +
        "Cerebras_Docs + examples + source-code.\n\n" +
        "Results are grouped by layer (knowledge → docs → examples → source).\n\n" +
        "**Search rule**: Prefer recall over precision — better to return extra results than miss relevant content.",
      inputSchema: {
        type: "object" as const,
        properties: {
          query: {
            type: "string",
            description: "Search query (supports grep regex)",
          },
          file_pattern: {
            type: "string",
            description:
              'File glob filter applied to code layers (default: none, searches all). Use "*.csl" or "*.py" to narrow.',
          },
          max_results: {
            type: "number",
            description: "Max matches per layer (default 15)",
          },
        },
        required: ["query"],
      },
    },
    {
      name: "get_wse_differences",
      description:
        "Get WSE-2 vs WSE-3 architecture differences.\n\n" +
        "**When to use**: When the user's question involves hardware-specific behavior, " +
        "migration between WSE generations, or when you need to verify which features " +
        "are WSE-3 only (microthreads, @initialize_queue, etc.).\n\n" +
        "Optionally filter by feature keyword.",
      inputSchema: {
        type: "object" as const,
        properties: {
          feature: {
            type: "string",
            description:
              'Optional feature keyword to filter (e.g. "queue", "microthread", "task_id")',
          },
        },
      },
    },
  ],
}));

// ---------------------------------------------------------------------------
// tools/call
// ---------------------------------------------------------------------------

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  switch (name) {
    // ---- search_docs ----
    case "search_docs": {
      const query = args?.query as string;
      const filePattern = (args?.file_pattern as string) || "*.md";
      const maxResults = (args?.max_results as number) || 30;
      const ignoreCase = (args?.ignore_case as boolean) ?? true;

      const result = grepSearch(query, DOCS_DIR, {
        includePattern: filePattern,
        maxResults,
        ignoreCase,
      });
      return {
        content: [{ type: "text", text: result.replaceAll(KB_ROOT + "/", "") }],
      };
    }

    // ---- search_code ----
    case "search_code": {
      const query = args?.query as string;
      const scope = (args?.scope as string) || "all";
      const filePattern = (args?.file_pattern as string) || "*.csl";
      const maxResults = (args?.max_results as number) || 30;

      const searchPaths: string[] = [];
      if (scope === "examples" || scope === "all") searchPaths.push(EXAMPLES_DIR);
      if (scope === "libraries" || scope === "all") searchPaths.push(SOURCE_DIR);

      const results = searchPaths
        .map((p) =>
          grepSearch(query, p, { includePattern: filePattern, maxResults })
        )
        .join("\n\n");

      return {
        content: [
          { type: "text", text: results.replaceAll(KB_ROOT + "/", "") },
        ],
      };
    }

    // ---- search_knowledge (D3: enhanced with doc summaries) ----
    case "search_knowledge": {
      const query = args?.query as string;
      const maxResults = (args?.max_results as number) || 30;

      if (!fs.existsSync(KNOWLEDGE_DIR)) {
        return {
          content: [
            { type: "text", text: "AI-Knowledge-Base/ directory not found." },
          ],
        };
      }

      // List available knowledge docs with first-line summaries (D3)
      const files = fs
        .readdirSync(KNOWLEDGE_DIR)
        .filter((f) => f.endsWith(".md"));
      const docList = files
        .map((f) => {
          const summary = getFirstSummaryLine(path.join(KNOWLEDGE_DIR, f));
          return summary ? `- ${f}: ${summary}` : `- ${f}`;
        })
        .join("\n");

      let header =
        `Available knowledge docs (${files.length}):\n${docList}\n\n` +
        `--- Search results for "${query}" ---\n\n`;

      const result = grepSearch(query, KNOWLEDGE_DIR, {
        includePattern: "*.md",
        maxResults,
        ignoreCase: true,
        contextLines: 3,
      });

      return {
        content: [
          {
            type: "text",
            text: header + result.replaceAll(KB_ROOT + "/", ""),
          },
        ],
      };
    }

    // ---- read_file ----
    case "read_file": {
      const filePath = resolvePath(args?.path as string);
      const maxLines = args?.max_lines as number | undefined;

      if (!fs.existsSync(filePath)) {
        return {
          content: [{ type: "text", text: `File not found: ${args?.path}` }],
        };
      }

      const stat = fs.statSync(filePath);

      if (stat.isDirectory()) {
        const entries = fs.readdirSync(filePath, { withFileTypes: true });
        const listing = entries
          .map((e) => (e.isDirectory() ? e.name + "/" : e.name))
          .join("\n");
        return {
          content: [
            { type: "text", text: `Directory: ${args?.path}\n\n${listing}` },
          ],
        };
      }

      let content = fs.readFileSync(filePath, "utf-8");

      if (maxLines) {
        const lines = content.split("\n");
        if (lines.length > maxLines) {
          content =
            lines.slice(0, maxLines).join("\n") +
            `\n\n... (showing first ${maxLines} of ${lines.length} lines)`;
        }
      } else if (content.length > 100_000) {
        content =
          content.slice(0, 100_000) +
          "\n\n... (truncated at 100K characters)";
      }

      return { content: [{ type: "text", text: content }] };
    }

    // ---- list_examples ----
    case "list_examples": {
      const keyword = (args?.keyword as string)?.toLowerCase();
      const type = (args?.type as string) || "all";
      const parts: string[] = [];

      if (type === "tutorials" || type === "all") {
        const tutPath = path.join(EXAMPLES_DIR, "tutorials");
        if (fs.existsSync(tutPath)) {
          let dirs = fs
            .readdirSync(tutPath, { withFileTypes: true })
            .filter((d) => d.isDirectory())
            .map((d) => d.name);
          if (keyword) dirs = dirs.filter((d) => d.toLowerCase().includes(keyword));

          parts.push(`## Tutorials (${dirs.length})`);
          for (const d of dirs) {
            const files = fs.readdirSync(path.join(tutPath, d));
            parts.push(`- ${d}/  [${files.join(", ")}]`);
          }
        }
      }

      if (type === "benchmarks" || type === "all") {
        const benchPath = path.join(EXAMPLES_DIR, "benchmarks");
        if (fs.existsSync(benchPath)) {
          let dirs = fs
            .readdirSync(benchPath, { withFileTypes: true })
            .filter((d) => d.isDirectory())
            .map((d) => d.name);
          if (keyword)
            dirs = dirs.filter((d) => d.toLowerCase().includes(keyword));

          parts.push(`\n## Benchmarks (${dirs.length})`);
          for (const d of dirs) {
            parts.push(`- ${d}/`);
          }
        }
      }

      return {
        content: [
          { type: "text", text: parts.join("\n") || "No examples found." },
        ],
      };
    }

    // ---- lookup_feature (D2: appended multi-layer guidance) ----
    case "lookup_feature": {
      const feature = (args?.feature as string).toLowerCase();

      if (!fs.existsSync(CLAUDE_MD)) {
        return {
          content: [{ type: "text", text: "CLAUDE.md index not found." }],
        };
      }

      const claudeMd = fs.readFileSync(CLAUDE_MD, "utf-8");
      const lines = claudeMd.split("\n");

      // Extract matching table rows with their section headers
      const matches: string[] = [];
      let currentSection = "";
      let tableHeader = ""; // the row before |---|

      for (let i = 0; i < lines.length; i++) {
        const line = lines[i];

        if (line.startsWith("## ") || line.startsWith("### ")) {
          currentSection = line;
          tableHeader = "";
        }

        // Capture table header row (the row before |---|)
        if (
          i + 1 < lines.length &&
          lines[i + 1].match(/^\|[\s-|]+\|$/)
        ) {
          tableHeader = line;
        }

        // Match table content rows containing the feature keyword
        if (
          line.includes("|") &&
          !line.match(/^\|[\s-|]+\|$/) &&
          line.toLowerCase().includes(feature)
        ) {
          if (
            matches.length === 0 ||
            !matches.includes(currentSection)
          ) {
            matches.push("");
            matches.push(currentSection);
            if (tableHeader && tableHeader !== currentSection) {
              matches.push(tableHeader);
              const cols = tableHeader.split("|").length;
              matches.push(
                "|" + Array(cols - 2).fill("---").join("|") + "|"
              );
            }
          }
          matches.push(line);
        }
      }

      // D2: Append multi-layer search guidance
      const guidance =
        "\n\n---\n" +
        "**Next steps for comprehensive coverage:**\n" +
        "1. `search_knowledge` — Check AI-curated summaries for this topic\n" +
        "2. `search_docs` — Search official documentation for specification details\n" +
        "3. `search_code` — Find concrete implementation examples\n" +
        "Do not stop at one layer if the answer may be incomplete.";

      if (matches.length === 0) {
        // Fallback: search docs for the keyword
        const grepResult = grepSearch(feature, DOCS_DIR, {
          includePattern: "*.md",
          maxResults: 15,
          contextLines: 0,
          ignoreCase: true,
        });
        return {
          content: [
            {
              type: "text",
              text:
                `No exact index matches for "${feature}". Searching docs...\n\n` +
                grepResult.replaceAll(KB_ROOT + "/", "") +
                guidance,
            },
          ],
        };
      }

      return {
        content: [{ type: "text", text: matches.join("\n") + guidance }],
      };
    }

    // ---- get_topic_guide (D4: new tool) ----
    case "get_topic_guide": {
      const topic = (args?.topic as string).toLowerCase();
      const sections: string[] = [];
      sections.push(`# Topic Guide: "${topic}"\n`);
      sections.push(
        "Follow these layers in order for comprehensive coverage:\n"
      );

      // Layer 1: AI-Knowledge-Base
      sections.push("## Layer 1: AI-Knowledge-Base (curated summaries)");
      if (fs.existsSync(KNOWLEDGE_DIR)) {
        const kbFiles = fs
          .readdirSync(KNOWLEDGE_DIR)
          .filter((f) => f.endsWith(".md"));
        const nameMatches = kbFiles.filter((f) =>
          f.toLowerCase().includes(topic)
        );
        // Also grep for topic in all KB files
        const grepMatches = grepSearch(topic, KNOWLEDGE_DIR, {
          includePattern: "*.md",
          maxResults: 10,
          contextLines: 0,
          ignoreCase: true,
        });
        if (nameMatches.length > 0) {
          sections.push(
            `Directly relevant docs: ${nameMatches.map((f) => `AI-Knowledge-Base/${f}`).join(", ")}`
          );
        }
        if (grepMatches && grepMatches !== "No matches found.") {
          const mentionedFiles = new Set<string>();
          for (const line of grepMatches.split("\n")) {
            const m = line.match(/^([^:]+):/);
            if (m)
              mentionedFiles.add(m[1].replace(KNOWLEDGE_DIR + "/", ""));
          }
          const otherFiles = [...mentionedFiles].filter(
            (f) => !nameMatches.includes(f)
          );
          if (otherFiles.length > 0) {
            sections.push(
              `Also mentioned in: ${otherFiles.map((f) => `AI-Knowledge-Base/${f}`).join(", ")}`
            );
          }
        }
        if (nameMatches.length === 0 && grepMatches === "No matches found.") {
          sections.push("No knowledge base matches found.");
        }
      }
      sections.push("");

      // Layer 2-5: Extract from CLAUDE.md index
      if (fs.existsSync(CLAUDE_MD)) {
        const claudeMd = fs.readFileSync(CLAUDE_MD, "utf-8");
        const docPaths = new Set<string>();
        const tutPaths = new Set<string>();
        const benchPaths = new Set<string>();
        const libPaths = new Set<string>();

        for (const line of claudeMd.split("\n")) {
          if (!line.includes("|") || !line.toLowerCase().includes(topic))
            continue;
          // Extract paths by prefix
          const pathMatches = line.match(/`([^`]+)`/g);
          if (pathMatches) {
            for (const p of pathMatches) {
              const clean = p.replace(/`/g, "");
              if (clean.startsWith("doc/")) docPaths.add(clean);
              else if (clean.startsWith("tut/")) tutPaths.add(clean);
              else if (clean.startsWith("bench/")) benchPaths.add(clean);
              else if (clean.startsWith("lib/")) libPaths.add(clean);
            }
          }
        }

        sections.push("## Layer 2: Official Documentation (Cerebras_Docs/)");
        if (docPaths.size > 0) {
          for (const p of docPaths) sections.push(`- ${p}`);
        } else {
          // Fallback: grep docs
          const gr = grepSearch(topic, DOCS_DIR, {
            includePattern: "*.md",
            maxResults: 8,
            contextLines: 0,
            ignoreCase: true,
          });
          if (gr && gr !== "No matches found.") {
            const files = new Set<string>();
            for (const line of gr.split("\n")) {
              const m = line.match(/^([^:]+):/);
              if (m) files.add(m[1].replace(KB_ROOT + "/", ""));
            }
            for (const f of files) sections.push(`- ${f}`);
          } else {
            sections.push("No direct doc matches.");
          }
        }
        sections.push("");

        sections.push("## Layer 3: Tutorials (examples/tutorials/)");
        if (tutPaths.size > 0) {
          for (const p of tutPaths) sections.push(`- ${p}`);
        } else {
          sections.push(
            "No tutorial matches in index. Try: list_examples with keyword."
          );
        }
        sections.push("");

        sections.push("## Layer 4: Benchmarks (examples/benchmarks/)");
        if (benchPaths.size > 0) {
          for (const p of benchPaths) sections.push(`- ${p}`);
        } else {
          sections.push(
            "No benchmark matches in index. Try: list_examples with keyword."
          );
        }
        sections.push("");

        sections.push(
          "## Layer 5: Library Source (source-code/csl_libraries/)"
        );
        if (libPaths.size > 0) {
          for (const p of libPaths) sections.push(`- ${p}`);
        } else {
          // Fallback: grep source
          const gr = grepSearch(topic, SOURCE_DIR, {
            includePattern: "*.csl",
            maxResults: 5,
            contextLines: 0,
            ignoreCase: true,
          });
          if (gr && gr !== "No matches found.") {
            const files = new Set<string>();
            for (const line of gr.split("\n")) {
              const m = line.match(/^([^:]+):/);
              if (m) files.add(m[1].replace(KB_ROOT + "/", ""));
            }
            for (const f of files) sections.push(`- ${f}`);
          } else {
            sections.push("No library source matches.");
          }
        }
      }

      return { content: [{ type: "text", text: sections.join("\n") }] };
    }

    // ---- search_all (D5 + D11: cross-layer search with grouping) ----
    case "search_all": {
      const query = args?.query as string;
      const maxResults = (args?.max_results as number) || 15;
      const filePattern = args?.file_pattern as string | undefined;
      const parts: string[] = [];

      // Layer 1: Knowledge base (always *.md)
      const kbResult = grepSearch(query, KNOWLEDGE_DIR, {
        includePattern: "*.md",
        maxResults,
        ignoreCase: true,
        contextLines: 1,
      });
      if (kbResult && kbResult !== "No matches found.") {
        parts.push("=== AI-Knowledge-Base (curated summaries) ===");
        parts.push(kbResult.replaceAll(KB_ROOT + "/", ""));
        parts.push("");
      }

      // Layer 2: Official docs (always *.md)
      const docsResult = grepSearch(query, DOCS_DIR, {
        includePattern: "*.md",
        maxResults,
        ignoreCase: true,
        contextLines: 1,
      });
      if (docsResult && docsResult !== "No matches found.") {
        parts.push("=== Cerebras_Docs (official documentation) ===");
        parts.push(docsResult.replaceAll(KB_ROOT + "/", ""));
        parts.push("");
      }

      // Layer 3: Examples (uses file_pattern if provided)
      const codeResult = grepSearch(query, EXAMPLES_DIR, {
        includePattern: filePattern,
        maxResults,
        ignoreCase: true,
        contextLines: 1,
      });
      if (codeResult && codeResult !== "No matches found.") {
        parts.push("=== examples/ (tutorials + benchmarks) ===");
        parts.push(codeResult.replaceAll(KB_ROOT + "/", ""));
        parts.push("");
      }

      // Layer 4: Source code (uses file_pattern if provided)
      const srcResult = grepSearch(query, SOURCE_DIR, {
        includePattern: filePattern,
        maxResults,
        ignoreCase: true,
        contextLines: 1,
      });
      if (srcResult && srcResult !== "No matches found.") {
        parts.push("=== source-code/ (library source) ===");
        parts.push(srcResult.replaceAll(KB_ROOT + "/", ""));
        parts.push("");
      }

      const text =
        parts.length > 0
          ? parts.join("\n")
          : `No matches found for "${query}" in any layer.`;

      return { content: [{ type: "text", text }] };
    }

    // ---- get_wse_differences (D6: new tool) ----
    case "get_wse_differences": {
      const feature = (args?.feature as string | undefined)?.toLowerCase();

      // Read from CLAUDE.md WSE-2 vs WSE-3 table
      let wseSection = "";
      if (fs.existsSync(CLAUDE_MD)) {
        const claudeMd = fs.readFileSync(CLAUDE_MD, "utf-8");
        const wseStart = claudeMd.indexOf("## WSE-2 vs WSE-3");
        if (wseStart !== -1) {
          const afterStart = claudeMd.slice(wseStart);
          const nextSection = afterStart.indexOf("\n## ", 1);
          wseSection =
            nextSection !== -1
              ? afterStart.slice(0, nextSection)
              : afterStart;
        }
      }

      // Also check migration guide
      const migrationGuide = path.join(
        KNOWLEDGE_DIR,
        "CSL_WSE2_vs_WSE3_Migration_Guide.md"
      );
      let migrationContent = "";
      if (fs.existsSync(migrationGuide)) {
        migrationContent = fs.readFileSync(migrationGuide, "utf-8");
      }

      let result = "";

      if (feature) {
        // Filter to relevant content
        const filteredParts: string[] = [];

        if (wseSection) {
          const filtered = wseSection
            .split("\n")
            .filter(
              (line) =>
                line.startsWith("## ") ||
                line.startsWith("| Feature") ||
                line.startsWith("|---") ||
                line.toLowerCase().includes(feature)
            )
            .join("\n");
          if (filtered.includes("|")) {
            filteredParts.push("### From CLAUDE.md Quick Reference:\n");
            filteredParts.push(filtered);
          }
        }

        if (migrationContent) {
          const mLines = migrationContent.split("\n");
          const relevant: string[] = [];
          let inSection = false;

          for (let i = 0; i < mLines.length; i++) {
            const line = mLines[i];
            if (line.startsWith("## ") || line.startsWith("### ")) {
              inSection = line.toLowerCase().includes(feature);
              if (inSection) relevant.push(line);
            } else if (inSection) {
              relevant.push(line);
            } else if (line.toLowerCase().includes(feature)) {
              const start = Math.max(0, i - 2);
              const end = Math.min(mLines.length, i + 3);
              relevant.push("...");
              for (let j = start; j < end; j++) relevant.push(mLines[j]);
            }
          }

          if (relevant.length > 0) {
            filteredParts.push("\n### From Migration Guide:\n");
            filteredParts.push(relevant.join("\n"));
          }
        }

        result =
          filteredParts.length > 0
            ? filteredParts.join("\n")
            : `No specific WSE differences found for "${feature}". Full comparison:\n\n${wseSection}`;
      } else {
        result = wseSection;
        if (migrationContent) {
          result +=
            "\n\n---\nFull migration guide: AI-Knowledge-Base/CSL_WSE2_vs_WSE3_Migration_Guide.md\n" +
            "Use read_file to access the complete guide.";
        }
      }

      return {
        content: [
          {
            type: "text",
            text: result || "WSE comparison data not found.",
          },
        ],
      };
    }

    default:
      throw new McpError(ErrorCode.MethodNotFound, `Unknown tool: ${name}`);
  }
});

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error(`CSL Knowledge Server v2.0.0 running — KB root: ${KB_ROOT}`);
  console.error(`9 tools, 2 resources, server instructions loaded.`);
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
