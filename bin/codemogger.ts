#!/usr/bin/env bun
import { program } from "commander"
import { CodeIndex, type SearchMode } from "../src/index.ts"
import { localEmbed, LOCAL_MODEL_NAME } from "../src/embed/local.ts"
import { formatJson } from "../src/format/json.ts"
import { formatText } from "../src/format/text.ts"

program
  .name("codemogger")
  .description("Code indexing library for AI coding agents — semantic search over codebases")
  .version("0.2.0")
  .option("--db <path>", "database file path")

program
  .command("index")
  .description("Index a directory of source code")
  .argument("<dir>", "directory to index")
  .option("--language <lang>", "filter by language (e.g. rust, typescript)")
  .option("--verbose", "show detailed indexing progress")
  .action(async (dir: string, opts: { language?: string; verbose?: boolean }) => {
    const db = new CodeIndex({ dbPath: program.opts().db, embedder: localEmbed, embeddingModel: LOCAL_MODEL_NAME })
    try {
      const result = await db.index(dir, {
        languages: opts.language ? [opts.language] : undefined,
        verbose: opts.verbose,
      })

      if (opts.verbose || result.errors.length > 0) {
        for (const err of result.errors) {
          console.error(`warning: ${err}`)
        }
      }

      console.log(
        `Indexed ${result.files} file${result.files !== 1 ? "s" : ""} → ` +
        `${result.chunks} chunks, ` +
        `embedded ${result.embedded}, ` +
        `skipped ${result.skipped} unchanged, ` +
        `removed ${result.removed} stale ` +
        `(${result.duration}ms)`
      )
    } finally {
      await db.close()
    }
  })

program
  .command("search")
  .description("Search indexed code semantically")
  .argument("<query>", "natural language query or search terms")
  .option("--limit <n>", "maximum results to return", "5")
  .option("--threshold <score>", "minimum score to include", "0")
  .option("--format <fmt>", "output format: json|text", "json")
  .option("--snippet", "include code snippet in output")
  .option("--mode <mode>", "search mode: semantic|keyword|hybrid", "semantic")
  .action(async (query: string, opts: { limit: string; threshold: string; format: string; snippet?: boolean; mode: string }) => {
    const db = new CodeIndex({ dbPath: program.opts().db, embedder: localEmbed, embeddingModel: LOCAL_MODEL_NAME })
    try {
      const start = performance.now()
      const results = await db.search(query, {
        limit: parseInt(opts.limit, 10),
        threshold: parseFloat(opts.threshold),
        includeSnippet: opts.snippet,
        mode: opts.mode as SearchMode,
      })
      const elapsed = Math.round(performance.now() - start)

      switch (opts.format) {
        case "text":
          console.log(formatText(query, results, elapsed))
          break
        case "json":
        default:
          console.log(formatJson(query, results, elapsed))
          break
      }
    } finally {
      await db.close()
    }
  })

program
  .command("list")
  .description("List all indexed files")
  .option("--format <fmt>", "output format: json|text", "text")
  .action(async (opts: { format: string }) => {
    const db = new CodeIndex({ dbPath: program.opts().db, embedder: localEmbed, embeddingModel: LOCAL_MODEL_NAME })
    try {
      const files = await db.listFiles()

      if (opts.format === "json") {
        console.log(JSON.stringify(files, null, 2))
      } else {
        if (files.length === 0) {
          console.log("No files indexed. Run `codemogger index <dir>` first.")
          return
        }
        const totalChunks = files.reduce((sum, f) => sum + f.chunkCount, 0)
        console.log(`${files.length} file${files.length !== 1 ? "s" : ""} indexed (${totalChunks} chunks):\n`)
        for (const f of files) {
          console.log(`  ${f.filePath} (${f.chunkCount} chunks)`)
        }
      }
    } finally {
      await db.close()
    }
  })

program.parse()
