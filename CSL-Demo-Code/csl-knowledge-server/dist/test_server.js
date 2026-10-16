#!/usr/bin/env node
/**
 * Smoke tests for csl-knowledge-server MCP tools.
 *
 * Usage:
 *   node --experimental-strip-types csl-knowledge-server/src/test_server.ts
 *   # or after build:
 *   node csl-knowledge-server/dist/test_server.js
 *
 * Tests each tool with known queries and validates results.
 */
import { execSync } from "node:child_process";
import * as path from "node:path";
const KB_ROOT = path.resolve(import.meta.dirname, "../..");
const SERVER_SCRIPT = path.join(import.meta.dirname, "index.ts");
let passed = 0;
let failed = 0;
function assert(condition, message) {
    if (condition) {
        passed++;
        console.log(`  ✓ ${message}`);
    }
    else {
        failed++;
        console.error(`  ✗ ${message}`);
    }
}
/**
 * Send a JSON-RPC request to the server via stdin and parse the response.
 * Uses a simple subprocess approach: pipe two requests (initialize + tool call).
 */
function callTool(toolName, args) {
    const initReq = JSON.stringify({
        jsonrpc: "2.0",
        id: 1,
        method: "initialize",
        params: {
            protocolVersion: "2024-11-05",
            capabilities: {},
            clientInfo: { name: "test", version: "1.0.0" },
        },
    });
    const initNotif = JSON.stringify({
        jsonrpc: "2.0",
        method: "notifications/initialized",
        params: {},
    });
    const toolReq = JSON.stringify({
        jsonrpc: "2.0",
        id: 2,
        method: "tools/call",
        params: { name: toolName, arguments: args },
    });
    const input = `${initReq}\n${initNotif}\n${toolReq}\n`;
    try {
        const result = execSync(`echo '${input.replace(/'/g, "'\\''")}' | node --experimental-strip-types ${SERVER_SCRIPT} ${KB_ROOT}`, {
            encoding: "utf-8",
            maxBuffer: 5 * 1024 * 1024,
            timeout: 30000,
            env: { ...process.env, NODE_NO_WARNINGS: "1" },
        });
        // Parse the last JSON-RPC response (the tool call result)
        const lines = result.trim().split("\n");
        for (let i = lines.length - 1; i >= 0; i--) {
            try {
                const parsed = JSON.parse(lines[i]);
                if (parsed.id === 2 && parsed.result?.content) {
                    return parsed.result.content
                        .map((c) => c.text || "")
                        .join("\n");
                }
            }
            catch {
                continue;
            }
        }
        return "";
    }
    catch (e) {
        const err = e;
        console.error(`    Tool call failed: ${err.stderr || err.stdout || e}`);
        return "";
    }
}
// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
console.log("\n=== CSL Knowledge Server Smoke Tests ===\n");
// Test 1: search_docs
console.log("search_docs:");
{
    const result = callTool("search_docs", { query: "DSD" });
    assert(result.length > 0, "Returns results for 'DSD'");
    assert(result.includes("dsd") || result.includes("DSD"), "Contains DSD content");
}
// Test 2: search_code
console.log("search_code:");
{
    const result = callTool("search_code", { query: "@fmacs" });
    assert(result.length > 0, "Returns results for '@fmacs'");
    assert(result.includes("fmacs"), "Contains fmacs usage");
}
{
    const result = callTool("search_code", {
        query: "memcpy_h2d",
        file_pattern: "*.py",
    });
    assert(result.length > 0, "Returns .py results for 'memcpy_h2d'");
}
// Test 3: search_knowledge
console.log("search_knowledge:");
{
    const result = callTool("search_knowledge", { query: "async" });
    assert(result.includes("Available knowledge docs"), "Lists available docs");
    assert(result.length > 200, "Returns substantial results");
}
// Test 4: read_file
console.log("read_file:");
{
    const result = callTool("read_file", { path: "AI-Knowledge-Base" });
    assert(result.includes("Directory:"), "Lists directory contents");
}
{
    const result = callTool("read_file", { path: "CLAUDE.md", max_lines: 5 });
    assert(result.includes("CSL"), "Reads CLAUDE.md");
}
{
    const result = callTool("read_file", { path: "nonexistent/file.txt" });
    assert(result.includes("not found"), "Reports missing files");
}
// Test 5: list_examples
console.log("list_examples:");
{
    const result = callTool("list_examples", { keyword: "gemv" });
    assert(result.includes("gemv-00"), "Lists gemv tutorials");
}
{
    const result = callTool("list_examples", {
        keyword: "stencil",
        type: "benchmarks",
    });
    assert(result.includes("stencil"), "Lists stencil benchmarks");
}
// Test 6: lookup_feature
console.log("lookup_feature:");
{
    const result = callTool("lookup_feature", { feature: "DSD" });
    assert(result.includes("|"), "Returns table rows");
    assert(result.includes("Next steps for comprehensive coverage"), "Includes multi-layer guidance (D2)");
}
// Test 7: get_topic_guide (D4)
console.log("get_topic_guide:");
{
    const result = callTool("get_topic_guide", { topic: "FIFO" });
    assert(result.includes("Layer 1"), "Has Layer 1");
    assert(result.includes("Layer 2"), "Has Layer 2");
    assert(result.includes("Layer 5"), "Has Layer 5");
    assert(result.includes("AI-Knowledge-Base") || result.includes("Knowledge"), "References knowledge base");
}
// Test 8: search_all (D5)
console.log("search_all:");
{
    const result = callTool("search_all", { query: "@allocate_fifo" });
    assert(result.length > 0, "Returns cross-layer results");
    assert(result.includes("==="), "Groups results by layer (D11)");
}
// Test 9: get_wse_differences (D6)
console.log("get_wse_differences:");
{
    const result = callTool("get_wse_differences", {});
    assert(result.includes("WSE-2"), "Returns WSE comparison");
    assert(result.includes("WSE-3"), "Includes WSE-3 info");
}
{
    const result = callTool("get_wse_differences", { feature: "queue" });
    assert(result.toLowerCase().includes("queue"), "Filters by feature keyword");
}
// Edge cases
console.log("Edge cases:");
{
    const result = callTool("search_docs", { query: "xyznonexistent123" });
    assert(result.includes("No matches") || result.length === 0, "Handles no-match gracefully");
}
// Summary
console.log(`\n=== Results: ${passed} passed, ${failed} failed ===\n`);
process.exit(failed > 0 ? 1 : 0);
