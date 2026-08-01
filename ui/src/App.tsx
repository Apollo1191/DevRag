import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from "react";

type Chunk = {
  text: string;
  score: number;
  metadata: {
    relative_path?: string;
    start_line?: number;
    end_line?: number;
  };
};

type QueryResponse = {
  answer?: string | null;
  chunks: Chunk[];
};

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  chunks?: Chunk[];
};

type IndexItem = {
  store_path: string;
  total_chunks: number;
  sample: { relative_path?: string; start_line?: number; end_line?: number }[];
  label?: string | null;
  topics?: string[];
};

type IngestJobStatus = {
  status: string;
  stage?: string | null;
  message?: string | null;
  total_files?: number | null;
  total_chunks?: number | null;
  total_embeddings?: number | null;
  error?: string | null;
};

type ClearStorageResponse = {
  message: string;
  deleted_storage: boolean;
};

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export function App() {
  const [repoUrl, setRepoUrl] = useState("https://github.com/psf/requests");
  const [localRepoPath, setLocalRepoPath] = useState("/app/workspaces/httpx");
  const [ingestLoading, setIngestLoading] = useState(false);
  const [ingestMessage, setIngestMessage] = useState<string>("");

  const [question, setQuestion] = useState("How do I use Depends in FastAPI?");
  const [topK, setTopK] = useState(5);
  const [queryLoading, setQueryLoading] = useState(false);
  const [queryResult, setQueryResult] = useState<QueryResponse | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [useLlm, setUseLlm] = useState(false);
  const [indexLoading, setIndexLoading] = useState(false);
  const [error, setError] = useState<string>("");
  const [indexInfo, setIndexInfo] = useState<{ stores: IndexItem[] } | null>(null);
  const [jobStatus, setJobStatus] = useState<IngestJobStatus | null>(null);
  const [jobId, setJobId] = useState<string | null>(() => window.localStorage.getItem("devrag.lastJobId"));
  const [lastIngestKind, setLastIngestKind] = useState<"repo" | "local" | null>(null);

  const hasResults = useMemo(() => !!queryResult?.chunks?.length, [queryResult]);
  const repoPresets = [
    { label: "Small real repo: Requests", value: "https://github.com/psf/requests" },
    { label: "Tiny local sample", value: "./workspaces/sample_repo" },
    { label: "Medium repo: HTTPX", value: "https://github.com/encode/httpx" },
  ];

  async function refreshIndex() {
    setIndexLoading(true);
    try {
      const response = await fetch(`${API_BASE}/index_info`);
      if (!response.ok) {
        throw new Error(`Index request failed (${response.status})`);
      }
      setIndexInfo(await response.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load index info");
    } finally {
      setIndexLoading(false);
    }
  }

  useEffect(() => {
    refreshIndex();
  }, []);

  useEffect(() => {
    if (!jobId) {
      return;
    }

    const activeJobId = jobId;
    let active = true;
    let timer = 0;
    async function pollJob() {
      try {
        const res = await fetch(`${API_BASE}/ingest_status/${activeJobId}`);
        if (!res.ok) {
          if (res.status === 404) {
            setJobStatus({ status: "expired", stage: "expired", message: "This ingest job is no longer available on the backend." });
            window.clearInterval(timer);
          }
          return;
        }
        const body = (await res.json()) as IngestJobStatus;
        if (!active) {
          return;
        }
        setJobStatus(body);
        window.localStorage.setItem("devrag.lastJobId", activeJobId);
        if (body.status === "completed" || body.status === "failed") {
          window.clearInterval(timer);
          await refreshIndex();
        }
      } catch (e) {
        if (active) {
          setError(e instanceof Error ? e.message : "Unable to read ingest status");
        }
      }
    }

    timer = window.setInterval(pollJob, 1500);
    pollJob();

    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [jobId]);

  async function startIngest(kind: "repo" | "local") {
    const endpoint = kind === "repo" ? "ingest_async" : "ingest_local_async";
    const payload = kind === "repo" ? { repo_url: repoUrl } : { repo_path: localRepoPath };

    const response = await fetch(`${API_BASE}/${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const body = await response.text();
      throw new Error(body || "Ingest failed");
    }

    const body = await response.json();
    setJobId(body.job_id);
    window.localStorage.setItem("devrag.lastJobId", body.job_id);
    setIngestMessage(body.message || "Ingestion started.");
    setJobStatus({ status: "running", stage: "queued", message: "Ingestion queued" });
    setLastIngestKind(kind);
  }

  async function handleIngest(event: FormEvent) {
    event.preventDefault();
    setError("");
    setIngestMessage("");
    setJobStatus(null);
    setJobId(null);
    setIngestLoading(true);

    try {
      await startIngest("repo");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Ingest failed");
    } finally {
      setIngestLoading(false);
    }
  }

  async function handleIngestLocal(event: FormEvent) {
    event.preventDefault();
    setError("");
    setIngestMessage("");
    setJobStatus(null);
    setJobId(null);
    setIngestLoading(true);

    try {
      await startIngest("local");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Local ingest failed");
    } finally {
      setIngestLoading(false);
    }
  }

  async function handleClearStorageAndReindex() {
    setError("");
    setIngestMessage("");
    setQueryResult(null);
    setJobStatus(null);
    setJobId(null);
    setIngestLoading(true);

    try {
      const clearResponse = await fetch(`${API_BASE}/storage/clear`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      if (!clearResponse.ok) {
        const body = await clearResponse.text();
        throw new Error(body || "Clear storage failed");
      }

      const cleared = (await clearResponse.json()) as ClearStorageResponse;
      setIndexInfo(null);
      setIngestMessage(cleared.message);

      if (lastIngestKind === "local") {
        await startIngest("local");
      } else {
        await startIngest("repo");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Clear storage failed");
    } finally {
      setIngestLoading(false);
    }
  }

  async function handleQuery(event: FormEvent) {
    event.preventDefault();
    setError("");
    setQueryResult(null);
    setQueryLoading(true);

    try {
      const response = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: topK, use_llm: useLlm }),
      });

      if (!response.ok) {
        const body = await response.text();
        throw new Error(body || "Query failed");
      }

      const body = (await response.json()) as QueryResponse;
      setQueryResult(body);
      setChatMessages((messages) => [
        ...messages,
        { role: "user", content: question },
        ...(body.answer ? [{ role: "assistant" as const, content: body.answer, chunks: body.chunks }] : []),
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Query failed");
    } finally {
      setQueryLoading(false);
    }
  }

  return (
    <div className="page">
      <header className="hero">
        <p className="eyebrow">DevRag</p>
        <h1>Code Retrieval UI</h1>
        <p className="subtitle">Ingest a repo, ask a question, get grounded answers with source snippets.</p>
      </header>

      <section className="panel">
        <h2>1. Ingest Repository</h2>
        <form onSubmit={handleIngest} className="form">
          <label htmlFor="repo-url">Repository URL</label>
          <input
            id="repo-url"
            value={repoUrl}
            onChange={(event: ChangeEvent<HTMLInputElement>) => setRepoUrl(event.target.value)}
            placeholder="https://github.com/owner/repo"
          />
          <div className="preset-row">
            {repoPresets.map((preset) => (
              <button key={preset.label} type="button" onClick={() => setRepoUrl(preset.value)}>
                {preset.label}
              </button>
            ))}
          </div>
          <button type="submit" disabled={ingestLoading}>
            {ingestLoading ? "Indexing..." : "Ingest"}
          </button>
        </form>
        {ingestMessage ? <p className="success">{ingestMessage}</p> : null}
        <form onSubmit={handleIngestLocal} className="form" style={{ marginTop: 16 }}>
          <label htmlFor="local-repo-path">Or ingest local sample repo</label>
          <input
            id="local-repo-path"
            value={localRepoPath}
            onChange={(event: ChangeEvent<HTMLInputElement>) => setLocalRepoPath(event.target.value)}
            placeholder="/app/workspaces/httpx"
          />
          <button type="submit" disabled={ingestLoading}>
            {ingestLoading ? "Indexing..." : "Ingest Local Sample"}
          </button>
        </form>
        {jobStatus ? (
          <div className="status-box">
            <p>
              <strong>Status:</strong> {jobStatus.status}
            </p>
            <p>
              <strong>Stage:</strong> {jobStatus.stage || "unknown"}
            </p>
            <p>{jobStatus.message || ""}</p>
            {jobId ? <p><strong>Job ID:</strong> {jobId}</p> : null}
            {typeof jobStatus.total_files === "number" ? <p>Files: {jobStatus.total_files}</p> : null}
            {typeof jobStatus.total_chunks === "number" ? <p>Chunks: {jobStatus.total_chunks}</p> : null}
            {typeof jobStatus.total_embeddings === "number" ? <p>Embeddings: {jobStatus.total_embeddings}</p> : null}
            {jobStatus.error ? <p className="error">{jobStatus.error}</p> : null}
          </div>
        ) : null}
      </section>

      <section className="panel">
        <h2>Index Info</h2>
        <div className="toolbar">
          <button type="button" onClick={refreshIndex} disabled={indexLoading}>
            {indexLoading ? "Refreshing..." : "Refresh index"}
          </button>
          <span className="muted">API: {API_BASE}</span>
        </div>
        {indexInfo && indexInfo.stores && indexInfo.stores.length ? (
          <div>
            <div style={{ marginBottom: 12 }}>
              <button type="button" onClick={handleClearStorageAndReindex} disabled={ingestLoading}>
                {ingestLoading ? "Clearing..." : "Clear storage and reindex"}
              </button>
            </div>
            {indexInfo.stores.map((s: any, i: number) => (
              <article key={i} className="chunk">
                <header>
                  <strong>{s.label || s.store_path}</strong>
                  <span> total chunks: {s.total_chunks}</span>
                </header>
                {s.topics && s.topics.length ? (
                  <p>
                    <em>Repos/topics:</em> {s.topics.slice(0, 5).join(", ")}
                  </p>
                ) : null}
                <div>
                  <em>Sample sources:</em>
                  <ul>
                    {s.sample.map((it: any, idx: number) => (
                      <li key={idx}>{it.relative_path}:{it.start_line}-{it.end_line}</li>
                    ))}
                  </ul>
                </div>
              </article>
            ))}
          </div>
        ) : (
          <div>
            <p>No index available yet. Reingest a repo to refresh the summary.</p>
            <button type="button" onClick={handleClearStorageAndReindex} disabled={ingestLoading}>
              {ingestLoading ? "Working..." : "Clear storage and reindex"}
            </button>
          </div>
        )}
      </section>

      <section className="panel">
        <h2>2. Ask Question</h2>
        {chatMessages.length ? (
          <div className="chat-log" aria-live="polite">
            {chatMessages.map((message, index) => (
              <article className={`chat-message ${message.role}`} key={`${message.role}-${index}`}>
                <strong>{message.role === "user" ? "You" : "DevRag"}</strong>
                <p>{message.content}</p>
              </article>
            ))}
          </div>
        ) : null}
        <form onSubmit={handleQuery} className="form">
          <label htmlFor="question">Question</label>
          <textarea
            id="question"
            rows={4}
            value={question}
            onChange={(event: ChangeEvent<HTMLTextAreaElement>) => setQuestion(event.target.value)}
          />
          <label htmlFor="top-k">Top K</label>
          <input
            id="top-k"
            type="number"
            min={1}
            max={20}
            value={topK}
            onChange={(event: ChangeEvent<HTMLInputElement>) => setTopK(Number(event.target.value || 5))}
          />
          <label className="check-row" htmlFor="use-llm">
            <input
              id="use-llm"
              type="checkbox"
              checked={useLlm}
              onChange={(event: ChangeEvent<HTMLInputElement>) => setUseLlm(event.target.checked)}
            />
            Use OpenAI answer generation
          </label>
          <button type="submit" disabled={queryLoading}>
            {queryLoading ? "Searching..." : useLlm ? "Ask OpenAI" : "Search sources"}
          </button>
          {chatMessages.length ? (
            <button type="button" className="secondary-button" onClick={() => setChatMessages([])}>
              Clear chat
            </button>
          ) : null}
        </form>
      </section>

      {error ? <section className="panel error">{error}</section> : null}

      {queryResult?.answer ? (
        <section className="panel">
          <h2>Answer</h2>
          <p className="answer">{queryResult.answer}</p>
        </section>
      ) : null}

      {queryResult && !queryResult.answer ? (
        <section className="panel">
          <h2>Answer</h2>
          <p className="error">LLM is off or unavailable. Showing retrieval sources only.</p>
        </section>
      ) : null}

      {hasResults ? (
        <section className="panel">
          <h2>Sources</h2>
          <div className="results">
            {queryResult!.chunks.map((chunk, index) => (
              <article className="chunk" key={`${chunk.metadata.relative_path}-${index}`}>
                <header>
                  <strong>
                    {chunk.metadata.relative_path}:{chunk.metadata.start_line}-{chunk.metadata.end_line}
                  </strong>
                  <span>score {chunk.score.toFixed(3)}</span>
                </header>
                <pre>{chunk.text}</pre>
              </article>
            ))}
          </div>
        </section>
      ) : null}
    </div>
  );
}
