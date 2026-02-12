import { join, resolve } from "path"
import { mkdirSync, statSync, existsSync } from "fs"
import { Store, type SearchResult, type IndexedFile, type Codebase } from "./db/store.ts"
import { scanDirectory } from "./scan/walker.ts"
import { chunkFile } from "./chunk/treesitter.ts"
import { detectLanguage } from "./chunk/languages.ts"
import { preprocessQuery, type QueryMode } from "./search/query.ts"
import { rrfMerge } from "./search/rank.ts"
import type { Embedder } from "./embed/types.ts"

export type { SearchResult, IndexedFile, Codebase } from "./db/store.ts"
export type { CodeChunk } from "./chunk/types.ts"
export type { Embedder } from "./embed/types.ts"

export type SearchMode = "semantic" | "keyword" | "hybrid"

export interface SearchOptions {
  limit?: number
  threshold?: number
  includeSnippet?: boolean
  mode?: SearchMode
}

export interface IndexOptions {
  languages?: string[]
  verbose?: boolean
}

export interface IndexResult {
  files: number
  chunks: number
  embedded: number
  skipped: number
  removed: number
  errors: string[]
  duration: number
}

export interface CodeIndexOptions {
  dbPath?: string
  /** Embedding function - SDK users must provide their own */
  embedder: Embedder
  /** Model name stored alongside embeddings (e.g. "all-MiniLM-L6-v2") */
  embeddingModel: string
}

function defaultDbPath(): string {
  const home = process.env.HOME ?? process.env.USERPROFILE ?? "."
  const dir = join(home, ".config", "codemogger")
  mkdirSync(dir, { recursive: true })
  return join(dir, "codemogger.db")
}

export class CodeIndex {
  private store: Store | null = null
  private dbPath: string
  private embedder: Embedder
  private embeddingModel: string
  private searchVerified = false

  constructor(opts: CodeIndexOptions) {
    this.dbPath = opts.dbPath ?? defaultDbPath()
    this.embedder = opts.embedder
    this.embeddingModel = opts.embeddingModel
  }

  private async getStore(): Promise<Store> {
    if (!this.store) {
      this.store = await Store.open(this.dbPath)
    }
    return this.store
  }

  /** Index a directory: scan files, chunk with tree-sitter, embed, store */
  async index(dir: string, opts?: IndexOptions): Promise<IndexResult> {
    const start = performance.now()
    const store = await this.getStore()
    const rootDir = resolve(dir)

    // Get or create codebase entry
    const codebaseId = await store.getOrCreateCodebase(rootDir)

    // Phase 1: Scan directory for source files
    const t0 = performance.now()
    const { files, errors } = await scanDirectory(rootDir, opts?.languages)
    const scanTime = Math.round(performance.now() - t0)

    let filesProcessed = 0
    let chunksCreated = 0
    let skipped = 0
    const activeFiles = new Set<string>()

    // Phase 2: Check hashes, chunk changed files, embed — pipelined in batches
    const t1 = performance.now()

    // Batch hash lookups: check which files changed (using absolute paths)
    const filesToProcess: typeof files = []
    for (const file of files) {
      activeFiles.add(file.absPath)
      const storedHash = await store.getFileHash(codebaseId, file.absPath)
      if (storedHash === file.hash) {
        skipped++
      } else {
        filesToProcess.push(file)
      }
    }

    // Build embedding text for a chunk
    function buildEmbedText(s: { filePath: string; kind: string; name: string; signature: string; snippet: string }): string {
      let text = s.filePath
      if (s.kind && s.name) text += `: ${s.kind} ${s.name}`
      else if (s.name) text += `: ${s.name}`
      if (s.signature) text += `\n${s.signature}`
      if (s.snippet) {
        const preview = s.snippet.slice(0, 500)
        text += `\n${preview}`
      }
      return text
    }

    // Process in streaming batches: chunk → write → embed per batch
    const FILE_BATCH = 200
    const EMBED_BATCH = 64
    let embedded = 0

    for (let batchStart = 0; batchStart < filesToProcess.length; batchStart += FILE_BATCH) {
      const batchFiles = filesToProcess.slice(batchStart, batchStart + FILE_BATCH)

      // Chunk this batch
      const batchChunks: { filePath: string; fileHash: string; chunks: import("./chunk/types.ts").CodeChunk[] }[] = []
      for (const file of batchFiles) {
        const langConfig = detectLanguage(file.absPath)
        if (!langConfig) continue
        try {
          const chunks = await chunkFile(file.absPath, file.content, file.hash, langConfig)
          batchChunks.push({ filePath: file.absPath, fileHash: file.hash, chunks })
          filesProcessed++
          chunksCreated += chunks.length
        } catch (e: any) {
          errors.push({ path: file.absPath, error: e.message ?? String(e) })
        }
      }

      // Write chunks to DB
      if (batchChunks.length > 0) {
        await store.batchUpsertAllFileChunks(codebaseId, batchChunks)
      }

      // Embed this batch
      const stale = await store.getStaleEmbeddings(codebaseId, this.embeddingModel)
      if (stale.length > 0) {
        for (let i = 0; i < stale.length; i += EMBED_BATCH) {
          const slice = stale.slice(i, i + EMBED_BATCH)
          const texts = slice.map(buildEmbedText)
          const vectors = await this.embedder(texts)
          await store.batchUpsertEmbeddings(
            slice.map((s, j) => ({
              chunkKey: s.chunkKey,
              embedding: vectors[j],
              modelName: this.embeddingModel,
            }))
          )
          embedded += vectors.length
        }
      }
    }

    const chunkAndEmbedTime = Math.round(performance.now() - t1)

    // Phase 3: Remove chunks for deleted files
    const removed = await store.removeStaleFiles(codebaseId, activeFiles)

    // Phase 5: Build per-codebase FTS table
    const t3 = performance.now()
    await store.rebuildFtsTable(codebaseId)
    const ftsTime = Math.round(performance.now() - t3)

    // Update codebase timestamp
    await store.touchCodebase(codebaseId)

    const duration = Math.round(performance.now() - start)

    // Log phase timing if verbose
    if (opts?.verbose) {
      console.log(`  scan: ${scanTime}ms, chunk+embed: ${chunkAndEmbedTime}ms (${embedded} chunks), fts: ${ftsTime}ms`)
    }
    return {
      files: filesProcessed,
      chunks: chunksCreated,
      embedded,
      skipped,
      removed,
      errors,
      duration,
    }
  }

  /** Search for code chunks relevant to a query.
   *  - "semantic": natural language / conceptual queries (vector search, global)
   *  - "keyword": precise identifier or term lookup (FTS, queries all codebases)
   *  - "hybrid": combine both via reciprocal rank fusion
   */
  async search(query: string, opts?: SearchOptions): Promise<SearchResult[]> {
    const store = await this.getStore()
    const limit = opts?.limit ?? 5
    const threshold = opts?.threshold ?? 0.0
    const includeSnippet = opts?.includeSnippet ?? false
    const mode = opts?.mode ?? "semantic"

    // Verify the DB is in a readable state before searching
    await this.verifySearchable(store)

    if (mode === "semantic") {
      const [queryVec] = await this.embedder([query])
      const results = await store.vectorSearch(queryVec, limit, includeSnippet)
      return threshold > 0 ? results.filter((r) => r.score >= threshold) : results
    }

    // Keyword path: preprocess query for FTS
    const processed = preprocessQuery(query, "keywords")
    if (!processed.trim()) return []

    const ftsResults = await store.ftsSearch(processed, limit, includeSnippet)

    if (mode === "keyword") {
      return threshold > 0 ? ftsResults.filter((r) => r.score >= threshold) : ftsResults
    }

    // Hybrid: combine keyword + semantic via RRF
    const [queryVec] = await this.embedder([query])
    const vecResults = await store.vectorSearch(queryVec, limit, includeSnippet)
    const merged = rrfMerge(ftsResults, vecResults, limit)
    return threshold > 0 ? merged.filter((r) => r.score >= threshold) : merged
  }

  /**
   * Verify the database is in a searchable state (once per instance).
   * Detects when the DB file is large but has no visible chunks
   * (e.g., WAL locked by another process, missing WAL file).
   */
  private async verifySearchable(store: Store): Promise<void> {
    if (this.searchVerified) return
    this.searchVerified = true

    // Quick check: does the DB file look like it should have data?
    if (!existsSync(this.dbPath)) return
    const fileSize = statSync(this.dbPath).size
    if (fileSize <= 1_000_000) return // small/empty DB, nothing to verify

    // DB file is >1MB — verify at least one codebase has chunks
    const codebases = await store.listCodebases()
    if (codebases.length === 0) return // no codebases registered, fine

    // Check if any codebase reports having chunks (from indexed_files, fast)
    const totalChunks = codebases.reduce((sum, c) => sum + c.chunkCount, 0)
    if (totalChunks > 0) return // indexed_files metadata says we have chunks, good

    throw new Error(
      `Database file is ${(fileSize / 1e6).toFixed(0)}MB but contains no indexed chunks. ` +
      `The database may be locked by another process, or the WAL file may be missing or inaccessible.`
    )
  }

  /** List all indexed files (optionally scoped to a directory/codebase) */
  async listFiles(): Promise<IndexedFile[]> {
    const store = await this.getStore()
    return store.listFiles()
  }

  /** List all codebases */
  async listCodebases(): Promise<Codebase[]> {
    const store = await this.getStore()
    return store.listCodebases()
  }

  /** Close the database connection */
  async close(): Promise<void> {
    if (this.store) {
      this.store.close()
      this.store = null
    }
  }
}
