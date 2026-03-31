const { useState, useRef } = React;

const UI_SETTINGS_KEY = 'resume-scorer-ui-settings';
const DEFAULT_UI_SETTINGS = {
  stage1Threshold: 0.30,
  stage2Threshold: 0.30,
  stage3Threshold: 0.30,
  stage1Mode: 'skill_match',
  useLlm: true,
  embeddingProfile: 'openai-small',
};

function loadUiSettings() {
  try {
    const raw = window.localStorage.getItem(UI_SETTINGS_KEY);
    if (!raw) return DEFAULT_UI_SETTINGS;
    const parsed = JSON.parse(raw);
    return {
      ...DEFAULT_UI_SETTINGS,
      ...(parsed && typeof parsed === 'object' ? parsed : {}),
    };
  } catch (_) {
    return DEFAULT_UI_SETTINGS;
  }
}

/* ─── Helpers ──────────────────────────────────────────────────────────── */

function fmtScore(v) {
  return typeof v === 'number' ? Math.max(0, Math.min(1, v)).toFixed(2) : '—';
}

function fmtBytes(b) {
  if (b < 1024) return `${b}B`;
  if (b < 1048576) return `${(b / 1024).toFixed(1)}KB`;
  return `${(b / 1048576).toFixed(1)}MB`;
}

function scoreColor(v) {
  if (v >= 0.65) return '#10B981';
  if (v >= 0.35) return '#F59E0B';
  return '#EF4444';
}

function rankClass(n) {
  if (n === 1) return 'rank-1';
  if (n === 2) return 'rank-2';
  if (n === 3) return 'rank-3';
  return 'rank-n';
}

/* ─── Toggle pill ───────────────────────────────────────────────────────── */
function Toggle({ value, onChange }) {
  return (
    <div className="toggle-pill">
      {['Text', 'File'].map(opt => (
        <button
          key={opt}
          className={`toggle-opt ${value === opt ? 'active' : ''}`}
          onClick={() => onChange(opt)}
          type="button"
        >
          {opt}
        </button>
      ))}
    </div>
  );
}

/* ─── File drop zone ────────────────────────────────────────────────────── */
function DropZone({ accept, multiple, files, onChange, hint }) {
  const ref = useRef();
  const [dragging, setDragging] = useState(false);

  const add = (newFiles) => {
    const arr = [...newFiles];
    onChange(multiple ? [...files, ...arr] : [arr[0]]);
  };

  const remove = (i) => onChange(files.filter((_, idx) => idx !== i));

  return (
    <div>
      <div
        className={`drop-zone ${dragging ? 'drag-over' : ''}`}
        onClick={() => ref.current.click()}
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={e => { e.preventDefault(); setDragging(false); add([...e.dataTransfer.files]); }}
      >
        <div className="drop-icon">+</div>
        <div className="drop-text">
          <strong>Select files</strong> or drag &amp; drop<br />
          {hint}
        </div>
        <div className="drop-types">
          {accept.split(',').map(t => (
            <span key={t} className="type-tag">{t.trim()}</span>
          ))}
        </div>
      </div>
      <input
        ref={ref}
        type="file"
        accept={accept}
        multiple={multiple}
        style={{ display: 'none' }}
        onChange={e => { add([...e.target.files]); ref.current.value = ''; }}
      />
      {files.length > 0 && (
        <div className="file-list">
          {files.map((f, i) => (
            <div key={i} className="file-chip">
              <span className="file-chip-mark" />
              <span className="file-chip-name" title={f.name}>{f.name}</span>
              <span className="file-chip-size">{fmtBytes(f.size)}</span>
              <button
                className="file-chip-remove"
                type="button"
                onClick={e => { e.stopPropagation(); remove(i); }}
              >×</button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function SkillChip({ skill }) {
  return <span className="skill-chip" title={skill}>{skill}</span>;
}

/* ─── Score mini-bar ─────────────────────────────────────────────────────── */
function ScoreCell({ value, max = 1, rawTen = false }) {
  const v = value ?? 0;
  const width = rawTen ? Math.max(0, Math.min(100, (v / max) * 100)) : Math.max(0, Math.min(100, v * 100));
  const color = scoreColor(v);
  return (
    <div className="score-block">
      <span className="score-num" style={{ color }}>{fmtScore(v)}</span>
      <div className="score-track">
        <div className="score-fill" style={{ width: `${width}%`, background: color }} />
      </div>
    </div>
  );
}

/* ─── Status badge ───────────────────────────────────────────────────────── */
function StatusBadge({ stageEliminated }) {
  if (!stageEliminated) return <span className="badge badge-green">Passed</span>;
  const cls = stageEliminated === 1 ? 'badge-red' : stageEliminated === 2 ? 'badge-yellow' : 'badge-gray';
  return <span className={`badge ${cls}`}>Stage {stageEliminated}</span>;
}

/* ─── Expand panel ───────────────────────────────────────────────────────── */
function ExpandPanel({ result }) {
  const p = result.profile;
  const matched = result.key_skills || [];
  const missing = result.missing_skills || [];
  const hasSkills = matched.length > 0 || missing.length > 0;

  return (
    <div className="expand-content">

      {/* Skills gap */}
      {hasSkills && (
        <div style={{ marginBottom: 14 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text)', marginBottom: 8 }}>
            Skills match
          </div>
          {matched.length > 0 && (
            <div className="skills-section">
              <div className="skills-section-label" style={{ color: '#065F46' }}>Matched ({matched.length})</div>
              <div className="skills-row">
                {matched.map((s, i) => <span key={i} className="skill-chip">{s}</span>)}
              </div>
            </div>
          )}
          {missing.length > 0 && (
            <div className="skills-section">
              <div className="skills-section-label" style={{ color: '#991B1B' }}>Missing ({missing.length})</div>
              <div className="skills-row">
                {missing.map((s, i) => <span key={i} className="skill-chip-missing">{s}</span>)}
              </div>
            </div>
          )}
          {matched.length + missing.length > 0 && (
            <div style={{ marginTop: 6, height: 6, borderRadius: 3, background: 'var(--border)', overflow: 'hidden' }}>
              <div style={{
                height: '100%',
                width: `${Math.round(matched.length / (matched.length + missing.length) * 100)}%`,
                background: 'var(--success)',
                borderRadius: 3,
                transition: 'width 0.4s ease',
              }} />
            </div>
          )}
          {matched.length + missing.length > 0 && (
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
              {(matched.length / (matched.length + missing.length)).toFixed(2)} keyword coverage
            </div>
          )}
        </div>
      )}

      {/* LLM profile */}
      {p ? (
        <>
          <div className="verdict-cols">
            <div className="verdict-col col-strengths">
              <h5>Strengths</h5>
              <ul>{(p.strengths || []).map((s, i) => <li key={i}>{s}</li>)}</ul>
            </div>
            <div className="verdict-col col-weaknesses">
              <h5>Weaknesses</h5>
              <ul>{(p.weaknesses || []).map((s, i) => <li key={i}>{s}</li>)}</ul>
            </div>
            <div className="verdict-col col-unknowns">
              <h5>Unknowns</h5>
              <ul>{(p.unknowns || []).map((s, i) => <li key={i}>{s}</li>)}</ul>
            </div>
          </div>

          {p.values_evidence && (
            <div className="values-row">
              <strong>Values alignment: </strong>
              <span className={`align-${p.values_alignment}`}>{p.values_alignment}</span>
              {' · '}{p.values_evidence}
            </div>
          )}

          {p.verdict && <div className="verdict-box">{p.verdict}</div>}
        </>
      ) : (
        <div className="no-llm-note">
          Agent was not run for this candidate
          {result.stage_eliminated ? ` (eliminated at Stage ${result.stage_eliminated})` : ''}.
          Enable Agent and re-score to get strengths, weaknesses, and a hiring verdict.
        </div>
      )}
    </div>
  );
}

/* ─── Table row ──────────────────────────────────────────────────────────── */
function ResultRow({ result, expanded, onToggle, onJudge, judging }) {
  const ss = result.stage_scores || {};
  const hasProfile = !!result.profile;
  const canJudge = !hasProfile && !!result.resume_text;

  function RankCell() {
    if (judging) {
      return (
        <div className="rank-bubble rank-judging" title="Running LLM judge…">
          <div className="rank-spinner" />
        </div>
      );
    }
    if (canJudge) {
      return (
        <button
          type="button"
          className="judge-run-btn"
          title="Run LLM judge on this candidate"
          onClick={e => { e.stopPropagation(); onJudge(result); }}
        >
          Run Judge
        </button>
      );
    }
    return <div className={`rank-bubble ${rankClass(result.rank)}`}>{result.rank}</div>;
  }

  return (
    <>
      <tr className="data-row" onClick={onToggle}>
        <td><RankCell /></td>
        <td style={{ maxWidth: 160 }}>
          <div style={{ fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
               title={result.filename}>
            {result.filename}
          </div>
        </td>
        <td>
          <ScoreCell value={ss.bm25_score} />
        </td>
        <td>
          <ScoreCell value={ss.combined_score} />
        </td>
        <td>
          {ss.llm_score != null
            ? <ScoreCell value={ss.llm_score} />
            : <span style={{ color: 'var(--text-light)', fontSize: 11 }}>
                {canJudge ? <span style={{ color: 'var(--success)' }}>available</span> : '—'}
              </span>}
        </td>
        <td>
          <span style={{ fontWeight: 700, fontSize: 14, color: scoreColor(result.final_score) }}>
            {fmtScore(result.final_score)}
          </span>
        </td>
        <td><StatusBadge stageEliminated={result.stage_eliminated} /></td>
        <td>
          <span className="expand-icon">{expanded ? '▲' : '▼'}</span>
        </td>
      </tr>

      {expanded && (
        <tr className="expand-row">
          <td colSpan={8}>
            <ExpandPanel result={result} />
          </td>
        </tr>
      )}
    </>
  );
}

/* ─── Download CSV ───────────────────────────────────────────────────────── */
function downloadCSV(results) {
  const cols = ['Rank','Filename','Stage 1','Match','Agent','Final','Status','Matched Skills','Missing Skills','Verdict'];
  const rows = results.map(r => {
    const ss = r.stage_scores || {};
    return [
      r.rank,
      r.filename,
      fmtScore(ss.bm25_score),
      fmtScore(ss.combined_score),
      fmtScore(ss.llm_score),
      fmtScore(r.final_score),
      r.stage_eliminated ? `Stage ${r.stage_eliminated}` : 'Passed',
      (r.key_skills || []).join('; '),
      (r.missing_skills || []).join('; '),
      r.profile?.verdict ?? '',
    ];
  });
  const csv = [cols, ...rows]
    .map(row => row.map(c => `"${String(c ?? '').replace(/"/g, '""')}"`).join(','))
    .join('\n');
  const a = Object.assign(document.createElement('a'), {
    href: 'data:text/csv;charset=utf-8,' + encodeURIComponent(csv),
    download: 'resume_rankings.csv',
  });
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

/* ─── Results table ──────────────────────────────────────────────────────── */
function ResultsTable({ results, onJudge, judgingSet }) {
  const [expanded, setExpanded] = useState({});
  const toggle = fn => setExpanded(prev => ({ ...prev, [fn]: !prev[fn] }));

  return (
    <div className="table-wrap">
      <div className="table-header">
          <span className="table-title">
            Candidates
          <span style={{ marginLeft: 6, fontSize: 11, color: 'var(--text-muted)', fontWeight: 400 }}>
            Scores shown on a 0-1 scale
          </span>
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span className="table-count">{results.length} total</span>
          <button className="btn-download" onClick={() => downloadCSV(results)} type="button">
            ⬇ Download CSV
          </button>
        </div>
      </div>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Filename</th>
            <th>Stage 1</th>
            <th>Match</th>
            <th>Agent</th>
            <th>Final</th>
            <th>Status</th>
            <th>Details</th>
          </tr>
        </thead>
        <tbody>
          {results.map(r => (
            <ResultRow
              key={r.filename + r.rank}
              result={r}
              expanded={!!expanded[r.filename]}
              onToggle={() => toggle(r.filename)}
              onJudge={onJudge}
              judging={judgingSet.has(r.filename)}
            />
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ─── Main App ───────────────────────────────────────────────────────────── */
function App() {
  /* JD */
  const [jdMode, setJdMode]       = useState('Text');
  const [jdText, setJdText]       = useState('');
  const [jdFiles, setJdFiles]     = useState([]);

  /* Company values */
  const [valMode, setValMode]     = useState('Text');
  const [valText, setValText]     = useState('');
  const [valFiles, setValFiles]   = useState([]);

  /* Resumes */
  const [resFiles, setResFiles]   = useState([]);

  /* Internal settings loaded from a separate settings page */
  const initialSettings = loadUiSettings();
  const [s1] = useState(initialSettings.stage1Threshold);
  const [s2] = useState(initialSettings.stage2Threshold);
  const [s3] = useState(initialSettings.stage3Threshold);
  const [stage1Mode] = useState(initialSettings.stage1Mode || 'skill_match');
  const [llm] = useState(initialSettings.useLlm);
  const [embeddingProfile, setEmbeddingProfile] = useState(initialSettings.embeddingProfile);
  /* Scored results (mutable — updated by on-demand judge) */
  const [scoredResults, setScoredResults] = useState([]);
  const [responseStats, setResponseStats] = useState(null);
  const [lastJd, setLastJd]               = useState('');
  const [lastValues, setLastValues]       = useState('');
  const [judgingSet, setJudgingSet]       = useState(new Set());

  /* UI state */
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState('');

  async function handleScore() {
    setLoading(true);
    setError('');
    try {
      const fd = new FormData();

      if (jdMode === 'Text') {
        fd.append('job_description', jdText);
      } else {
        if (!jdFiles[0]) throw new Error('Please upload a JD file.');
        fd.append('jd_file', jdFiles[0]);
        fd.append('job_description', '');
      }

      if (valMode === 'Text') {
        fd.append('company_values', valText);
      } else {
        if (valFiles[0]) fd.append('values_file', valFiles[0]);
        fd.append('company_values', '');
      }

      if (!resFiles.length) throw new Error('Please upload at least one resume or ZIP archive.');
      resFiles.forEach(f => fd.append('resumes', f));

      fd.append('use_llm', llm ? 'true' : 'false');
      fd.append('stage1_threshold', s1);
      fd.append('stage2_threshold', s2);
      fd.append('stage3_threshold', s3);
      fd.append('stage1_mode', stage1Mode);
      fd.append('embedding_profile', embeddingProfile);
      fd.append('excluded_skills_json', JSON.stringify([]));

      const res = await fetch('/score/batch', { method: 'POST', body: fd });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setScoredResults(data.results || []);
      setResponseStats(data);
      setLastJd(data.job_description || '');
      setLastValues(data.company_values || '');
      setEmbeddingProfile(data.embedding_profile || embeddingProfile);
      setJudgingSet(new Set());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleJudge(result) {
    if (!result.resume_text) return;
    setJudgingSet(prev => new Set([...prev, result.filename]));
    try {
      const res = await fetch('/score/judge-single', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          resume_text: result.resume_text,
          resume_filename: result.filename,
          job_description: lastJd,
          company_values: lastValues,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const profile = await res.json();
      const llmScore = typeof profile.llm_score === 'number' ? profile.llm_score : null;
      const alpha = responseStats?.pipeline_stats?.alpha ?? 0.4;
      const beta = responseStats?.pipeline_stats?.beta ?? 0.6;
      const stage3Threshold = s3;
      setScoredResults(prev => prev.map(r =>
        r.filename === result.filename
          ? {
              ...r,
              profile,
              stage_scores: { ...r.stage_scores, llm_score: llmScore },
              final_score: llmScore != null
                ? parseFloat(((alpha * (r.stage_scores?.combined_score ?? 0)) + (beta * llmScore)).toFixed(4))
                : r.final_score,
            }
          : r
      ));
      setResponseStats(prev => {
        if (!prev) return prev;
        const alreadyJudged = result.stage_scores?.llm_score != null;
        const nextEvaluated = alreadyJudged ? prev.stage3_evaluated : prev.stage3_evaluated + 1;
        const wasSurvivor = (result.stage_scores?.llm_score ?? null) != null && !result.stage_eliminated && (result.stage_scores?.llm_score ?? 0) >= stage3Threshold;
        const isSurvivor = llmScore != null && !result.stage_eliminated && llmScore >= stage3Threshold;
        let nextSurvivors = prev.stage3_survivors;
        if (!wasSurvivor && isSurvivor) nextSurvivors += 1;
        if (wasSurvivor && !isSurvivor) nextSurvivors = Math.max(0, nextSurvivors - 1);
        return {
          ...prev,
          stage3_evaluated: nextEvaluated,
          stage3_survivors: nextSurvivors,
          pipeline_stats: {
            ...prev.pipeline_stats,
            stage3_evaluated: nextEvaluated,
            stage3_survivors: nextSurvivors,
          },
        };
      });
    } catch (e) {
      setError(`Judge failed for ${result.filename}: ${e.message}`);
    } finally {
      setJudgingSet(prev => { const s = new Set(prev); s.delete(result.filename); return s; });
    }
  }

  const canScore = (jdMode === 'Text' ? jdText.trim().length >= 50 : jdFiles.length > 0)
                   && resFiles.length > 0
                   && !loading;

  return (
    <div className="app-shell">
      {/* ── Navbar ── */}
      <nav className="navbar">
        <a className="navbar-brand" href="/">
          <div className="brand-icon">RS</div>
          <div>
            <div>Resume Scorer</div>
            <div className="brand-sub">Hiring ops workspace</div>
          </div>
        </a>
        <div className="navbar-right">
          <div className="nav-status">
            <span className="nav-status-dot" />
            Pipeline ready
          </div>
          <a className="nav-link" href="/docs" target="_blank" rel="noreferrer">
            API Docs ↗
          </a>
          <a className="nav-link" href="/eval" target="_blank" rel="noreferrer">
            Eval
          </a>
          <a className="nav-link" href="/settings" target="_blank" rel="noreferrer">
            Settings
          </a>
        </div>
      </nav>

      {/* ── Layout ── */}
      <div className="layout">

        {/* ── Left sidebar ── */}
        <div className="sidebar">
          <div className="sidebar-scroll">
            <div className="sidebar-intro">
              <div className="panel-kicker">Screening Console</div>
              <h1>Upload the role context and score resumes.</h1>
              <p>The main workflow stays focused on screening. Advanced scoring controls live on the separate settings page while this workspace stays centered on inputs, ranking, and review.</p>
            </div>

            {/* Job Description */}
            <div className="panel-card">
              <div className="section-title">
                <span>Job Description</span>
              </div>
              <Toggle value={jdMode} onChange={setJdMode} />
              {jdMode === 'Text'
                ? <textarea
                    className="input"
                    rows={6}
                    placeholder="Paste the job description here…"
                    value={jdText}
                    onChange={e => setJdText(e.target.value)}
                  />
                : <DropZone accept=".pdf,.txt" multiple={false} files={jdFiles}
                    onChange={setJdFiles} hint="Single PDF or TXT" />
              }
            </div>

            {/* Company Values */}
            <div className="panel-card">
              <div className="section-title">
                <span>Company Values</span>
                <span className="optional">(optional)</span>
              </div>
              <Toggle value={valMode} onChange={setValMode} />
              {valMode === 'Text'
                ? <textarea
                    className="input"
                    rows={4}
                    placeholder="Paste company values, culture, or mission…"
                    value={valText}
                    onChange={e => setValText(e.target.value)}
                  />
                : <DropZone accept=".pdf,.txt" multiple={false} files={valFiles}
                    onChange={setValFiles} hint="Single PDF or TXT" />
              }
            </div>

            {/* Resumes */}
            <div className="panel-card">
              <div className="section-title">
                <span>Resumes</span>
              </div>
              <DropZone
                accept=".pdf,.txt,.zip"
                multiple={true}
                files={resFiles}
                onChange={setResFiles}
                hint="Individual files or a ZIP archive"
              />
            </div>

          </div>

          {/* Score button */}
          <div className="sidebar-footer">
            <button
              className="btn-score"
              onClick={handleScore}
              disabled={!canScore}
              type="button"
            >
              {loading
                ? <><span style={{ width:16, height:16, border:'2px solid rgba(255,255,255,.3)', borderTopColor:'white', borderRadius:'50%', display:'inline-block', animation:'spin .75s linear infinite' }} /> Scoring…</>
                : <>Score Resumes</>
              }
            </button>
          </div>
        </div>

        {/* ── Results panel ── */}
        <div className="results-panel">
          <div className="results-hero">
            <div>
              <div className="panel-kicker">Candidate Review</div>
              <div className="results-hero-title">Ranked shortlist with filter-stage visibility</div>
              <div className="results-hero-copy">
                Compare skill match, semantic relevance, and LLM review in one place and drill into strengths, weaknesses, unknowns, and values alignment.
              </div>
            </div>
            <div className="results-hero-meta">
              <div className="hero-meta-card">
                <span>Role Type</span>
                <strong>{responseStats?.role_type || '—'}</strong>
              </div>
              <div className="hero-meta-card">
                <span>Agent</span>
                <strong>{llm ? 'Enabled' : 'Disabled'}</strong>
              </div>
              <div className="hero-meta-card">
                <span>Resumes</span>
                <strong>{resFiles.length}</strong>
              </div>
            </div>
          </div>

          {responseStats?.pipeline_stats?.semantic_scoring_summary ? (
            <div className="table-wrap" style={{ marginBottom: 2 }}>
              <div className="table-header">
                <span className="table-title">Semantic Scoring Input</span>
              </div>
              <div style={{ padding: '14px 18px 18px 18px', color: 'var(--text-muted)', fontSize: 12, lineHeight: 1.6 }}>
                <div style={{ marginBottom: 8, color: 'var(--text)' }}>
                  {responseStats.pipeline_stats.semantic_scoring_summary}
                </div>
                <div className="field-helper">JD preview used for semantic comparison</div>
                <div className="semantic-preview">{responseStats.pipeline_stats.semantic_query_preview || '—'}</div>
              </div>
            </div>
          ) : null}

          {error && (
            <div className="error-box"><span>{error}</span></div>
          )}

          {!responseStats && !loading && !error && (
            <div className="empty-state">
              <div className="empty-title">No results yet</div>
              <div className="empty-desc">
                Fill in the job description, upload resumes, then click <strong>Score Resumes</strong>.
              </div>
            </div>
          )}

          {loading && (
            <div className="empty-state">
              <div className="spinner" />
              <div className="empty-title">Scoring resumes…</div>
              <div className="empty-desc">
                Running skill match + semantic relevance
                {llm ? ', then Agent on survivors' : ''}.
                This may take a moment.
              </div>
            </div>
          )}

          {responseStats && !loading && (
            <>
              {/* Stats */}
              <div className="stats-grid">
                <div className="stat-card">
                  <div className="stat-val">{responseStats.total_resumes}</div>
                  <div className="stat-lbl">Total</div>
                </div>
                <div className="stat-card">
                  <div className="stat-val">{responseStats.stage1_survivors}</div>
                  <div className="stat-lbl">Skill Match Pass</div>
                </div>
                <div className="stat-card">
                  <div className="stat-val">{responseStats.stage2_survivors}</div>
                  <div className="stat-lbl">Stage 2 Pass</div>
                </div>
                <div className="stat-card">
                  <div className="stat-val">{responseStats.stage3_evaluated}</div>
                  <div className="stat-lbl">Agent Evaluated</div>
                </div>
                <div className="stat-card">
                  <div className="stat-val" style={{ color: '#10B981' }}>{responseStats.stage3_survivors}</div>
                  <div className="stat-lbl">Final Selected</div>
                </div>
              </div>

              {(responseStats.core_skills?.length || responseStats.adjacent_skills?.length || responseStats.soft_skills?.length || responseStats.value_keywords?.length) ? (
                <div className="table-wrap" style={{ marginBottom: 18 }}>
                  <div className="table-header">
                    <span className="table-title">Extracted Matching Signals</span>
                    <div />
                  </div>
                  {responseStats.core_skills?.length ? (
                    <div style={{ padding: '14px 18px 0 18px' }}>
                      <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 8 }}>JD Core Skills</div>
                      <div className="skills-row">
                        {responseStats.core_skills.map((s, i) => (
                          <SkillChip key={`core-${i}`} skill={s} />
                        ))}
                      </div>
                    </div>
                  ) : null}
                  {responseStats.adjacent_skills?.length ? (
                    <div style={{ padding: '14px 18px 0 18px' }}>
                      <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 8 }}>JD Adjacent Skills</div>
                      <div className="skills-row">
                        {responseStats.adjacent_skills.map((s, i) => (
                          <SkillChip key={`adjacent-${i}`} skill={s} />
                        ))}
                      </div>
                    </div>
                  ) : null}
                  {responseStats.soft_skills?.length ? (
                    <div style={{ padding: '14px 18px 0 18px' }}>
                      <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 8 }}>JD Soft Skills</div>
                      <div className="skills-row">
                        {responseStats.soft_skills.map((s, i) => (
                          <SkillChip key={`soft-${i}`} skill={s} />
                        ))}
                      </div>
                    </div>
                  ) : null}
                  {responseStats.value_keywords?.length ? (
                    <div style={{ padding: '14px 18px 18px 18px' }}>
                      <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 8 }}>Company Values Keywords</div>
                      <div className="skills-row">
                        {responseStats.value_keywords.map((s, i) => <span key={`val-${i}`} className="skill-chip">{s}</span>)}
                      </div>
                    </div>
                  ) : null}
                </div>
              ) : null}

              {/* Table */}
              {scoredResults.length > 0 && (
                <ResultsTable
                  results={scoredResults}
                  onJudge={handleJudge}
                  judgingSet={judgingSet}
                />
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
