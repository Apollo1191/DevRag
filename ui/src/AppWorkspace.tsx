import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type Chunk = {
  text: string;
  score: number;
  metadata: {
    relative_path?: string;
    start_line?: number;
    end_line?: number;
    source_repo?: string;
  };
};

type QueryResponse = { answer?: string | null; chunks: Chunk[] };
type ChatMessage = { role: "user" | "assistant"; content: string; chunks: Chunk[] };
type IndexItem = {
  store_path: string;
  total_chunks: number;
  sample: { relative_path?: string; start_line?: number; end_line?: number }[];
  label?: string | null;
  topics?: string[];
};
type JobStatus = {
  status: string;
  stage?: string | null;
  message?: string | null;
  total_files?: number | null;
  total_chunks?: number | null;
  total_embeddings?: number | null;
  error?: string | null;
};

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const QUICK_QUESTIONS = [
  "How does the HTTP client build and send a request?",
  "Where is timeout configuration applied?",
  "อธิบาย flow ของ authentication ใน repository นี้",
];

function MarkdownMessage({ content }: { content: string }) {
  return <div className="markdown-content">
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        code({ className, children, ...props }) {
          const language = className?.replace("language-", "") || "text";
          const code = String(children).replace(/\n$/, "");
          const isBlock = Boolean(className) || code.includes("\n");
          if (!isBlock) return <code className="inline-code" {...props}>{children}</code>;
          return <div className="code-block">
            <div className="code-toolbar"><span>{language}</span><button type="button" onClick={() => navigator.clipboard.writeText(code)}>Copy</button></div>
            <pre><code className={className} {...props}>{code}</code></pre>
          </div>;
        },
        a({ children, ...props }) { return <a {...props} target="_blank" rel="noreferrer">{children}</a>; },
      }}
    >{content}</ReactMarkdown>
  </div>;
}

export function AppWorkspace() {
  const [question, setQuestion] = useState(QUICK_QUESTIONS[0]);
  const [topK, setTopK] = useState(5);
  const [useLlm, setUseLlm] = useState(true);
  const [queryLoading, setQueryLoading] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [activeMessage, setActiveMessage] = useState<ChatMessage | null>(null);
  const [indexInfo, setIndexInfo] = useState<{ stores: IndexItem[] } | null>(null);
  const [indexLoading, setIndexLoading] = useState(false);
  const [repoUrl, setRepoUrl] = useState("https://github.com/psf/requests");
  const [localRepoPath, setLocalRepoPath] = useState("/app/workspaces/httpx");
  const [jobId, setJobId] = useState<string | null>(() => localStorage.getItem("devrag.lastJobId"));
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [ingestLoading, setIngestLoading] = useState(false);
  const [error, setError] = useState("");
  const [showIngest, setShowIngest] = useState(false);
  const conversationEndRef = useRef<HTMLDivElement>(null);

  const indexedRepos = indexInfo?.stores ?? [];
  const currentIndex = indexedRepos[0];
  const sourceCount = activeMessage?.chunks.length ?? 0;
  const sourceLabel = sourceCount ? `${sourceCount} sources attached` : "No evidence selected";
  const evidenceTone = sourceCount ? "ready" : "quiet";

  async function refreshIndex() {
    setIndexLoading(true);
    try {
      const response = await fetch(`${API_BASE}/index_info`);
      if (!response.ok) throw new Error(`Index request failed (${response.status})`);
      setIndexInfo(await response.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load index");
    } finally {
      setIndexLoading(false);
    }
  }

  useEffect(() => { refreshIndex(); }, []);

  useEffect(() => {
    conversationEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, queryLoading]);

  useEffect(() => {
    if (!jobId) return;
    let active = true;
    let timer = 0;
    const poll = async () => {
      try {
        const response = await fetch(`${API_BASE}/ingest_status/${jobId}`);
        if (!response.ok) {
          if (response.status === 404 && active) {
            setJobStatus({ status: "expired", stage: "expired", message: "This ingestion job is no longer available." });
            localStorage.removeItem("devrag.lastJobId");
            window.clearInterval(timer);
            setJobId(null);
          }
          return;
        }
        const body = (await response.json()) as JobStatus;
        if (!active) return;
        setJobStatus(body);
        if (body.status === "completed" || body.status === "failed") {
          window.clearInterval(timer);
          await refreshIndex();
          localStorage.removeItem("devrag.lastJobId");
          if (active) setJobId(null);
        }
      } catch (err) {
        if (active) setError(err instanceof Error ? err.message : "Unable to read job status");
      }
    };
    poll();
    timer = window.setInterval(poll, 2000);
    return () => { active = false; window.clearInterval(timer); };
  }, [jobId]);

  async function handleResetIndex() {
    if (!window.confirm("Reset the entire index? All ingested repositories will be removed.")) return;
    setError("");
    setIngestLoading(true);
    try {
      const response = await fetch(`${API_BASE}/storage/clear`, { method: "POST" });
      if (!response.ok) throw new Error(await response.text() || "Reset failed");
      setIndexInfo(null);
      setMessages([]);
      setActiveMessage(null);
      setJobStatus(null);
      setJobId(null);
      localStorage.removeItem("devrag.lastJobId");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Reset failed");
    } finally {
      setIngestLoading(false);
    }
  }

  async function startIngest(endpoint: "ingest_async" | "ingest_local_async", body: object) {
    const response = await fetch(`${API_BASE}/${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) throw new Error(await response.text() || "Ingest failed");
    const result = await response.json() as { job_id: string; message: string };
    setJobId(result.job_id);
    localStorage.setItem("devrag.lastJobId", result.job_id);
    setJobStatus({ status: "running", stage: "queued", message: result.message });
  }

  async function handleIngest(event: FormEvent) {
    event.preventDefault();
    setError("");
    setIngestLoading(true);
    try { await startIngest("ingest_async", { repo_url: repoUrl }); }
    catch (err) { setError(err instanceof Error ? err.message : "Ingest failed"); }
    finally { setIngestLoading(false); }
  }

  async function handleLocalIngest(event: FormEvent) {
    event.preventDefault();
    setError("");
    setIngestLoading(true);
    try { await startIngest("ingest_local_async", { repo_path: localRepoPath }); }
    catch (err) { setError(err instanceof Error ? err.message : "Local ingest failed"); }
    finally { setIngestLoading(false); }
  }

  async function handleQuery(event: FormEvent) {
    event.preventDefault();
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion || queryLoading) return;
    setError("");
    setQueryLoading(true);
    try {
      const response = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmedQuestion, top_k: topK, use_llm: useLlm }),
      });
      if (!response.ok) throw new Error(await response.text() || "Query failed");
      const result = await response.json() as QueryResponse;
      const userMessage: ChatMessage = { role: "user", content: trimmedQuestion, chunks: [] };
      const assistantMessage: ChatMessage = { role: "assistant", content: result.answer || "No answer returned.", chunks: result.chunks };
      setMessages((current) => [...current, userMessage, assistantMessage]);
      setActiveMessage(assistantMessage);
      setQuestion("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Query failed");
    } finally {
      setQueryLoading(false);
    }
  }

  const latestSources = useMemo(() => activeMessage?.chunks ?? [], [activeMessage]);

  return (
    <div className="workspace-shell">
      <aside className="sidebar">
        <div className="brand-lockup">
          <span className="brand-mark">D</span>
          <div><strong>DevRag</strong><small>Repository intelligence</small></div>
        </div>
        <div className="sidebar-section">
          <div className="section-kicker">Workspace</div>
          {indexedRepos.length ? indexedRepos.map((repo) => <div className="scope-card" key={repo.store_path}>
            <span className="status-dot" />
            <div><strong>{repo.label || "Repository"}</strong><small>{repo.total_chunks.toLocaleString()} chunks indexed</small></div>
          </div>) : <div className="scope-card"><span className="status-dot" /><div><strong>No repository</strong><small>Ingest a repository to begin</small></div></div>}
        </div>
        <div className="sidebar-section">
          <div className="section-kicker">Quick prompts</div>
          <div className="quick-list">
            {QUICK_QUESTIONS.map((prompt) => <button className="quick-prompt" key={prompt} onClick={() => setQuestion(prompt)}>{prompt}</button>)}
          </div>
        </div>
        <button className="sidebar-action" onClick={() => setShowIngest((value) => !value)}>{showIngest ? "Close ingest" : "Add repository"}</button>
        <div className="sidebar-footer"><small>API endpoint</small><span>{API_BASE}</span></div>
      </aside>

      <main className="chat-column">
        <header className="topbar">
          <div><span className="eyebrow">Codebase chat</span><h1>Understand the code behind the API.</h1></div>
          <div className="topbar-actions"><span className={`evidence-pill ${evidenceTone}`}>{sourceLabel}</span><button className="ghost-button" onClick={() => { setMessages([]); setActiveMessage(null); }}>New chat</button></div>
        </header>

        {showIngest ? <section className="ingest-drawer">
          <div className="drawer-heading"><div><span className="section-kicker">Index a codebase</span><h2>Bring a repository into the workspace</h2></div><button className="ghost-button" onClick={() => setShowIngest(false)}>Dismiss</button></div>
          <form className="ingest-grid" onSubmit={handleIngest}><label>Git repository<input value={repoUrl} onChange={(event: ChangeEvent<HTMLInputElement>) => setRepoUrl(event.target.value)} placeholder="https://github.com/owner/repo" /></label><button disabled={ingestLoading}>{ingestLoading ? "Indexing..." : "Index repository"}</button></form>
          <form className="ingest-grid local-form" onSubmit={handleLocalIngest}><label>Container path<input value={localRepoPath} onChange={(event: ChangeEvent<HTMLInputElement>) => setLocalRepoPath(event.target.value)} /></label><button className="secondary-button" disabled={ingestLoading}>{ingestLoading ? "Indexing..." : "Index local path"}</button></form>
          {jobStatus ? <div className={`job-strip ${jobStatus.status}`}><strong>{jobStatus.status}</strong><span>{jobStatus.stage || "working"}</span><span>{jobStatus.message}</span>{jobStatus.total_chunks ? <span>{jobStatus.total_chunks.toLocaleString()} chunks</span> : null}</div> : null}
          <div className="drawer-footer"><span>Reset removes all repositories from Qdrant.</span><button type="button" className="danger-button" onClick={handleResetIndex} disabled={ingestLoading}>Reset index</button></div>
        </section> : null}

        <section className="conversation" aria-live="polite">
          {!messages.length ? <div className="empty-state"><div className="empty-mark">↗</div><span className="section-kicker">Ask the repository</span><h2>Trace behavior. Find the right file. Learn the system.</h2><p>DevRag answers from indexed source code and keeps the evidence attached to every response.</p></div> : null}
          {messages.map((message, index) => <article className={`message-row ${message.role}`} key={`${message.role}-${index}`}><div className="avatar">{message.role === "user" ? "You" : "D"}</div><div className="message-body"><div className="message-label">{message.role === "user" ? "You" : "DevRag"}</div>{message.role === "assistant" ? <MarkdownMessage content={message.content} /> : <div className="message-content">{message.content}</div>}{message.role === "assistant" && message.chunks.length ? <button className="evidence-link" onClick={() => setActiveMessage(message)}>View {message.chunks.length} evidence sources →</button> : null}</div></article>)}
          {queryLoading ? <article className="message-row assistant"><div className="avatar">D</div><div className="message-body"><div className="message-label">DevRag</div><div className="typing-bubble"><span /><span /><span /></div></div></article> : null}
          <div ref={conversationEndRef} />
        </section>

        <form className="composer" onSubmit={handleQuery}><textarea value={question} onChange={(event: ChangeEvent<HTMLTextAreaElement>) => setQuestion(event.target.value)} placeholder="Ask about a function, request flow, exception, or file..." rows={3} /><div className="composer-footer"><div className="mode-switch"><button type="button" className={!useLlm ? "selected" : ""} onClick={() => setUseLlm(false)}>Sources only</button><button type="button" className={useLlm ? "selected" : ""} onClick={() => setUseLlm(true)}>Answer with OpenAI</button></div><div className="composer-right"><label>Top K <select value={topK} onChange={(event) => setTopK(Number(event.target.value))}>{[3, 5, 8, 10].map((value) => <option key={value} value={value}>{value}</option>)}</select></label><button className="send-button" disabled={queryLoading || !question.trim()}>{queryLoading ? "Searching..." : "Ask DevRag"}<span>↗</span></button></div></div></form>
        {error ? <div className="error-banner">{error}</div> : null}
      </main>

      <aside className="evidence-panel">
        <div className="evidence-header"><div><span className="section-kicker">Evidence</span><h2>Source inspector</h2></div><button className="icon-button" title="Refresh index" onClick={refreshIndex} disabled={indexLoading}>↻</button></div>
        {activeMessage ? <div className="evidence-meta"><span className="status-dot" />{latestSources.length} sources attached to last answer</div> : <div className="evidence-empty"><span>⌁</span><p>Sources from your latest answer will appear here.</p></div>}
        <div className="source-stack">{latestSources.map((chunk, index) => <article className="source-card" key={`${chunk.metadata.relative_path}-${index}`}><div className="source-card-header"><span>{String(index + 1).padStart(2, "0")}</span><strong>{chunk.metadata.relative_path || "unknown file"}</strong><em>{chunk.score.toFixed(2)}</em></div><div className="source-lines">L{chunk.metadata.start_line || "?"} - L{chunk.metadata.end_line || "?"}</div><pre>{chunk.text}</pre></article>)}</div>
        {indexedRepos.length ? <div className="index-summary"><span className="section-kicker">Indexed workspace</span>{indexedRepos.map((repo) => <div className="index-repo" key={repo.store_path}><strong>{repo.label || "Repository"}</strong><span>{repo.total_chunks.toLocaleString()} chunks</span></div>)}<button className="text-button" onClick={refreshIndex}>{indexLoading ? "Refreshing..." : "Refresh index"}</button></div> : null}
      </aside>
    </div>
  );
}
