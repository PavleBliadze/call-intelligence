const { useState, useRef, useEffect } = React;

// Two mirrored palettes: the dark theme is the exact same UI, just dark.
const THEMES = {
  light: {
    pageBg: '#FAFAFA',
    surface: '#ffffff',
    surfaceAlt: '#fcfcfc',
    ink: '#111111',
    sub: '#71717A',
    mut: '#a1a1aa',
    track: '#d4d4d8',
    detail: '#3f3f46',
    border: '#E4E4E7',
    soft: '#F4F4F5',
    red: '#E11020',
    redSoft: '#fdf2f3',
    redBorder: '#f1c0c4',
    topbar: '#111111',
    topbarText: '#ffffff',
    topbarSub: '#a1a1aa',
    topbarDivider: '#3f3f46',
    topbarBorder: 'none',
    codeBg: '#111111',
    codeText: '#e4e4e7',
  },
  dark: {
    pageBg: '#0b0b0d',
    surface: '#18181b',
    surfaceAlt: '#26262c',
    ink: '#f4f4f5',
    sub: '#b6b6bf',
    mut: '#8b8b95',
    track: '#52525b',
    detail: '#e4e4e7',
    border: '#3a3a41',
    soft: '#2c2c32',
    red: '#fb3a4a',
    redSoft: '#321519',
    redBorder: '#7a3036',
    topbar: '#161619',
    topbarText: '#f4f4f5',
    topbarSub: '#b6b6bf',
    topbarDivider: '#52525b',
    topbarBorder: '1px solid #3a3a41',
    codeBg: '#000000',
    codeText: '#e4e4e7',
  },
};
// Model-written text often uses em/en dashes; show them as a plain short dash.
function plainDashes(s) {
  return (s || '').replace(/\s*[\u2014\u2013]\s*/g, ' - ');
}
function fmtSize(b) {
  if (b >= 1e6) return (b / 1e6).toFixed(1) + ' MB';
  if (b >= 1e3) return Math.round(b / 1e3) + ' KB';
  return b + ' B';
}

function SunIcon() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="4"></circle>
      <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"></path>
    </svg>
  );
}
function MoonIcon() {
  return (
    <svg
      width="17"
      height="17"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"></path>
    </svg>
  );
}

function TopBar({ T, theme, onToggle }) {
  return (
    <header
      style={{
        background: T.topbar,
        color: T.topbarText,
        height: 56,
        flex: '0 0 56px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 24px',
        borderBottom: T.topbarBorder,
        transition: 'background .2s ease,color .2s ease',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 11 }}>
        <span style={{ fontSize: 15, fontWeight: 600, letterSpacing: '-0.01em' }}>
          Call Intelligence
        </span>
        <span style={{ width: 1, height: 16, background: T.topbarDivider }}></span>
        <span style={{ fontSize: 13, color: T.topbarSub }}>Console</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <button
          className="theme-toggle"
          onClick={onToggle}
          title={theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
          aria-label="Toggle theme"
        >
          {theme === 'light' ? <MoonIcon /> : <SunIcon />}
        </button>
        <span style={{ width: 1, height: 16, background: T.topbarDivider }}></span>
        <svg
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          stroke={T.red}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M12 3 4.5 6v5.5c0 4.6 3.2 8.4 7.5 9.5 4.3-1.1 7.5-4.9 7.5-9.5V6L12 3Z"></path>
        </svg>
        <span style={{ fontSize: 13, fontWeight: 600 }}>AI Hackathon</span>
      </div>
    </header>
  );
}

function StatusBadge({ T, status }) {
  if (status === 'suspicious')
    return (
      <span
        style={{
          fontSize: 11.5,
          fontWeight: 600,
          color: '#fff',
          background: T.red,
          padding: '4px 11px',
          borderRadius: 99,
          flex: '0 0 auto',
        }}
      >
        Suspicious
      </span>
    );
  if (status === 'safe')
    return (
      <span
        style={{
          fontSize: 11.5,
          fontWeight: 600,
          color: T.ink,
          background: T.soft,
          border: '1px solid ' + T.border,
          padding: '4px 11px',
          borderRadius: 99,
          flex: '0 0 auto',
        }}
      >
        Safe
      </span>
    );
  return (
    <span
      style={{
        fontSize: 11.5,
        fontWeight: 600,
        color: T.sub,
        background: T.surface,
        border: '1px solid ' + T.track,
        padding: '4px 11px',
        borderRadius: 99,
        flex: '0 0 auto',
      }}
    >
      Not analyzed
    </span>
  );
}

function FlagChip({ T, name }) {
  return (
    <span
      style={{
        fontSize: 11,
        fontWeight: 600,
        color: T.red,
        border: '1px solid ' + T.redBorder,
        background: T.redSoft,
        padding: '2px 8px',
        borderRadius: 6,
        whiteSpace: 'nowrap',
      }}
    >
      {name}
    </span>
  );
}

function TranscriptSection({ T, transcript, open, onToggle }) {
  if (!transcript) return null;
  return (
    <div>
      <button
        onClick={onToggle}
        aria-expanded={open}
        style={{
          background: 'none',
          border: 'none',
          color: T.sub,
          fontSize: 12.5,
          fontWeight: 500,
          cursor: 'pointer',
          padding: '14px 0 0',
          display: 'flex',
          alignItems: 'center',
          gap: 6,
        }}
      >
        <span
          style={{
            display: 'inline-flex',
            transform: 'rotate(' + (open ? 90 : 0) + 'deg)',
            transition: 'transform .2s',
          }}
        >
          <svg
            width="13"
            height="13"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="m9 18 6-6-6-6"></path>
          </svg>
        </span>
        {open ? 'Hide transcript' : 'View transcript'}
      </button>
      {open && (
        <pre
          style={{
            margin: '10px 0 0',
            padding: 14,
            background: T.codeBg,
            color: T.codeText,
            borderRadius: 10,
            fontFamily: "'Noto Sans Georgian',ui-monospace,monospace",
            fontSize: 12.5,
            lineHeight: 1.7,
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}
        >
          {plainDashes(transcript)}
        </pre>
      )}
    </div>
  );
}

function ResultCard({ T, r, open, onToggle, tOpen, onToggleT, analysisView = false }) {
  const flags = r.status === 'suspicious' ? r.triggered_rules || [] : [];
  // Skipped recordings were not analysed: shown dimmed, not as a result.
  const skipped = r.status === 'skipped';
  return (
    <div
      style={{
        border: '1px solid ' + T.border,
        borderRadius: 14,
        background: T.surface,
        overflow: 'hidden',
        opacity: skipped ? 0.55 : 1,
      }}
    >
      <div
        className="recording-heading"
        role="button"
        tabIndex={0}
        aria-expanded={open}
        aria-label={'Details for ' + r.file}
        onClick={onToggle}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            onToggle();
          }
        }}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 14,
          padding: '15px 18px',
          cursor: 'pointer',
        }}
      >
        <div
          style={{
            flex: '1 1 auto',
            minWidth: 0,
            fontSize: 13.5,
            fontWeight: 500,
            color: T.ink,
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}
        >
          {r.file}
        </div>
        {flags.length > 0 && (
          <div
            style={{
              flex: '0 1 auto',
              display: 'flex',
              flexWrap: 'wrap',
              justifyContent: 'flex-end',
              gap: 6,
            }}
          >
            {flags.map((x, i) => (
              <FlagChip key={i} T={T} name={x.name} />
            ))}
          </div>
        )}
        <StatusBadge T={T} status={r.status} />
        <span
          style={{
            flex: '0 0 16px',
            display: 'flex',
            alignItems: 'center',
            transform: 'rotate(' + (open ? 180 : 0) + 'deg)',
            transition: 'transform .2s',
          }}
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke={T.mut}
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="m6 9 6 6 6-6"></path>
          </svg>
        </span>
      </div>

      <div style={{ padding: '0 18px 16px' }}>
        {r.retry_error && (
          <p role="status" style={{ fontSize: 13, color: T.red, margin: '0 0 12px' }}>
            {r.retry_error} Showing the last successful analysis.
          </p>
        )}
        {r.summary && (
          <p style={{ fontSize: 14, lineHeight: 1.65, color: T.detail, margin: '0 0 12px' }}>
            {plainDashes(r.summary)}
          </p>
        )}
        {r.match_reason && (
          <p style={{ fontSize: 13, color: T.sub }}>
            <strong>Why this matches:</strong> {plainDashes(r.match_reason)}
          </p>
        )}
        {r.evidence_quote && (
          <blockquote
            style={{
              margin: '12px 0',
              padding: '8px 14px',
              borderLeft: '3px solid ' + T.red,
              color: T.ink,
              lineHeight: 1.6,
            }}
          >
            &ldquo;{plainDashes(r.evidence_quote)}&rdquo;
          </blockquote>
        )}
        {r.audio_url && (
          <div className="player-row">
            <audio
              controls
              controlsList={analysisView ? 'nodownload' : undefined}
              preload="none"
              src={r.audio_url}
              aria-label={'Play ' + r.file}
              style={{ width: '100%', maxWidth: 440, height: 38 }}
            />
            {!analysisView && (
              <a href={r.audio_url + '?download=1'} style={{ color: T.sub, fontSize: 12 }}>
                Download WAV
              </a>
            )}
            <span style={{ color: T.mut, fontSize: 12 }}>{Math.round(r.duration || 0)}s</span>
          </div>
        )}
        {analysisView && (
          <TranscriptSection T={T} transcript={r.transcript} open={tOpen} onToggle={onToggleT} />
        )}
      </div>

      {open && (
        <div style={{ borderTop: '1px solid ' + T.soft, padding: '6px 18px 18px' }}>
          {r.status === 'suspicious' && (
            <div>
              <div
                style={{
                  fontSize: 12,
                  fontWeight: 600,
                  color: T.sub,
                  letterSpacing: '0.03em',
                  textTransform: 'uppercase',
                  margin: '14px 0 10px',
                }}
              >
                Detected
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {(r.triggered_rules || []).map((x, i) => (
                  <div
                    key={i}
                    style={{
                      border: '1px solid ' + T.soft,
                      borderRadius: 10,
                      padding: '12px 14px',
                      background: T.surfaceAlt,
                    }}
                  >
                    <div style={{ fontSize: 13.5, fontWeight: 600, color: T.ink, marginBottom: 5 }}>
                      {x.name}
                    </div>
                    <div style={{ fontSize: 13.5, lineHeight: 1.55, color: T.detail }}>
                      {plainDashes(x.reason)}
                    </div>
                    {x.evidence_quote && (
                      <blockquote
                        style={{
                          margin: '10px 0 0',
                          paddingLeft: 12,
                          borderLeft: '2px solid ' + T.red,
                          color: T.ink,
                          fontSize: 13,
                          lineHeight: 1.6,
                        }}
                      >
                        {plainDashes(x.evidence_quote)}
                      </blockquote>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {skipped && (
            <div
              style={{
                display: 'flex',
                gap: 10,
                marginTop: 14,
                padding: '12px 14px',
                border: '1px solid ' + T.soft,
                borderRadius: 10,
                background: T.surfaceAlt,
              }}
            >
              <svg
                width="17"
                height="17"
                viewBox="0 0 24 24"
                fill="none"
                stroke={T.sub}
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
                style={{ flex: '0 0 17px', marginTop: 1 }}
              >
                <circle cx="12" cy="12" r="9"></circle>
                <path d="M12 8v5"></path>
                <path d="M12 16h.01"></path>
              </svg>
              <div style={{ fontSize: 13.5, lineHeight: 1.55, color: T.detail }}>
                {plainDashes(r.reason)}
              </div>
            </div>
          )}

          {r.status === 'safe' && (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                marginTop: 14,
                padding: '12px 14px',
                border: '1px solid ' + T.soft,
                borderRadius: 10,
                background: T.surfaceAlt,
              }}
            >
              <svg
                width="17"
                height="17"
                viewBox="0 0 24 24"
                fill="none"
                stroke={T.ink}
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                style={{ flex: '0 0 17px' }}
              >
                <path d="m5 12 4.5 4.5L19 7"></path>
              </svg>
              <div style={{ fontSize: 13.5, color: T.detail }}>
                No configured threat category detected.
              </div>
            </div>
          )}

          {!analysisView && (
            <TranscriptSection T={T} transcript={r.transcript} open={tOpen} onToggle={onToggleT} />
          )}
        </div>
      )}
    </div>
  );
}

function AnalyzePane({ T, theme, onUpdated }) {
  const [phase, setPhase] = useState('empty'); // empty | staged | results
  const [files, setFiles] = useState([]); // {id,file,name,size,status}
  const [results, setResults] = useState([]);
  const [dragOver, setDragOver] = useState(false);
  const [expanded, setExpanded] = useState({});
  const [showT, setShowT] = useState({});
  const idRef = useRef(0);

  const addFiles = (list) => {
    const incoming = Array.from(list || []).map((f) => ({
      id: 'f' + ++idRef.current,
      file: f,
      name: f.name,
      size: f.size,
      status: 'queued',
    }));
    if (!incoming.length) return;
    setFiles((prev) => [...prev, ...incoming]);
    setPhase('staged');
  };
  const onBrowse = (e) => {
    addFiles(e.target.files);
    e.target.value = '';
  };
  const onDragOver = (e) => {
    e.preventDefault();
    if (!dragOver) setDragOver(true);
  };
  const onDragLeave = (e) => {
    e.preventDefault();
    setDragOver(false);
  };
  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    addFiles(e.dataTransfer.files);
  };
  const removeFile = (id) =>
    setFiles((prev) => {
      const f = prev.filter((x) => x.id !== id);
      if (!f.length) setPhase('empty');
      return f;
    });
  const startOver = () => {
    setPhase('empty');
    setFiles([]);
    setResults([]);
    setExpanded({});
    setShowT({});
  };

  const runAnalysis = async () => {
    const batch = files;
    setPhase('results');
    setResults([]);
    setFiles((prev) => prev.map((f) => ({ ...f, status: 'queued' })));
    for (let i = 0; i < batch.length; i++) {
      setFiles((prev) => prev.map((f, idx) => (idx === i ? { ...f, status: 'analyzing' } : f)));
      let data;
      try {
        const fd = new FormData();
        fd.append('file', batch[i].file, batch[i].name);
        const res = await fetch('/api/analyze', { method: 'POST', body: fd });
        data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Analysis failed');
      } catch (e) {
        data = {
          file: batch[i].name,
          status: 'skipped',
          triggered_rules: [],
          transcript: '',
          reason: 'Upload failed: ' + e.message,
        };
      }
      data.uploadKey = batch[i].id;
      if (!data.id) data.id = batch[i].id;
      setResults((prev) => [...prev, data]);
      onUpdated();
      setFiles((prev) => prev.map((f, idx) => (idx === i ? { ...f, status: 'done' } : f)));
    }
  };

  const done = files.filter((f) => f.status === 'done').length;
  const progressPct = Math.round((done / (files.length || 1)) * 100) + '%';
  const running = phase === 'results' && done < files.length;
  const summary = results.reduce(
    (s, r) => {
      s[r.status] = (s[r.status] || 0) + 1;
      s.total++;
      return s;
    },
    { safe: 0, suspicious: 0, skipped: 0, total: 0 },
  );
  const analyzed = summary.total - summary.skipped;

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        background: T.pageBg,
        color: T.ink,
        transition: 'background .2s ease,color .2s ease',
      }}
    >
      <main
        style={{
          flex: '1 1 auto',
          display: 'flex',
          justifyContent: 'center',
          padding: '48px 24px 80px',
        }}
      >
        <div style={{ width: '100%', maxWidth: 900 }}>
          {phase === 'empty' && (
            <div>
              <div style={{ marginBottom: 28 }}>
                <h1
                  style={{
                    fontSize: 24,
                    fontWeight: 600,
                    letterSpacing: '-0.02em',
                    margin: '0 0 6px',
                  }}
                >
                  Analyze Phone Calls
                </h1>
                <p style={{ fontSize: 14, color: T.sub, margin: 0, lineHeight: 1.5 }}>
                  Analyze calls, inspect evidence, and save them to your searchable library. WAV
                  audio and transcripts are processed by OpenAI.
                </p>
              </div>
              <label
                htmlFor="fileInput"
                onDragOver={onDragOver}
                onDragLeave={onDragLeave}
                onDrop={onDrop}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 14,
                  minHeight: 280,
                  border: '2px dashed ' + (dragOver ? T.red : T.track),
                  borderRadius: 16,
                  background: dragOver ? T.redSoft : theme === 'light' ? '#eaeaec' : T.soft,
                  cursor: 'pointer',
                  transition: 'border-color .15s,background .15s',
                  textAlign: 'center',
                  padding: 40,
                }}
              >
                <span
                  style={{
                    width: 52,
                    height: 52,
                    borderRadius: 13,
                    background: T.surface,
                    border: '1px solid ' + T.border,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    boxShadow: '0 1px 2px rgba(0,0,0,.05)',
                  }}
                >
                  <svg
                    width="22"
                    height="22"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke={T.ink}
                    strokeWidth="1.6"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M12 16V4"></path>
                    <path d="m7 9 5-5 5 5"></path>
                    <path d="M5 16v2a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-2"></path>
                  </svg>
                </span>
                <div>
                  <div style={{ fontSize: 16, fontWeight: 500, marginBottom: 4, color: T.ink }}>
                    Drop audio files here or click to browse
                  </div>
                  <div style={{ fontSize: 13, color: T.sub }}>Accepts .wav files</div>
                </div>
              </label>
              <input
                id="fileInput"
                type="file"
                accept=".wav"
                multiple
                onChange={onBrowse}
                style={{ display: 'none' }}
              />
            </div>
          )}

          {phase === 'staged' && (
            <div>
              <div style={{ marginBottom: 18 }}>
                <h1
                  style={{
                    fontSize: 24,
                    fontWeight: 600,
                    letterSpacing: '-0.02em',
                    margin: '0 0 4px',
                  }}
                >
                  {files.length} file{files.length !== 1 ? 's' : ''} ready
                </h1>
                <p style={{ fontSize: 14, color: T.sub, margin: 0 }}>
                  Review the batch, then run the analysis.
                </p>
              </div>
              <div
                style={{
                  border: '1px solid ' + T.border,
                  borderRadius: 14,
                  background: T.surface,
                  overflow: 'hidden',
                }}
              >
                {files.map((f) => (
                  <div
                    key={f.id}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 12,
                      padding: '13px 16px',
                      borderBottom: '1px solid ' + T.soft,
                    }}
                  >
                    <span
                      style={{
                        width: 32,
                        height: 32,
                        borderRadius: 8,
                        background: T.soft,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        flex: '0 0 32px',
                      }}
                    >
                      <svg
                        width="15"
                        height="15"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke={T.sub}
                        strokeWidth="1.6"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <path d="M12 3v12"></path>
                        <path d="m8 11 4 4 4-4"></path>
                        <circle cx="12" cy="19" r="2"></circle>
                      </svg>
                    </span>
                    <div style={{ flex: '1 1 auto', minWidth: 0 }}>
                      <div
                        style={{
                          fontSize: 13.5,
                          fontWeight: 500,
                          color: T.ink,
                          whiteSpace: 'nowrap',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                        }}
                      >
                        {f.name}
                      </div>
                      <div style={{ fontSize: 12, color: T.mut, marginTop: 1 }}>
                        {fmtSize(f.size)}
                      </div>
                    </div>
                    <button
                      onClick={() => removeFile(f.id)}
                      title="Remove"
                      style={{
                        width: 28,
                        height: 28,
                        borderRadius: 7,
                        border: 'none',
                        background: 'none',
                        color: T.mut,
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        flex: '0 0 28px',
                        fontSize: 18,
                        lineHeight: 1,
                      }}
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 22 }}>
                <button
                  onClick={runAnalysis}
                  style={{
                    background: T.red,
                    color: '#fff',
                    border: 'none',
                    borderRadius: 10,
                    fontSize: 14.5,
                    fontWeight: 600,
                    padding: '13px 26px',
                    cursor: 'pointer',
                    boxShadow: '0 1px 2px rgba(225,16,32,.25)',
                  }}
                >
                  Run analysis
                </button>
                <label
                  htmlFor="fileInput2"
                  style={{
                    background: T.surface,
                    color: T.ink,
                    border: '1px solid ' + T.border,
                    borderRadius: 10,
                    fontSize: 14,
                    fontWeight: 500,
                    padding: '12px 18px',
                    cursor: 'pointer',
                  }}
                >
                  Add files
                </label>
                <input
                  id="fileInput2"
                  type="file"
                  accept=".wav"
                  multiple
                  onChange={onBrowse}
                  style={{ display: 'none' }}
                />
                <button
                  onClick={startOver}
                  style={{
                    background: 'none',
                    border: 'none',
                    color: T.sub,
                    fontSize: 13.5,
                    cursor: 'pointer',
                    padding: '12px 8px',
                    marginLeft: 'auto',
                  }}
                >
                  Clear
                </button>
              </div>
            </div>
          )}

          {phase === 'results' && (
            <div>
              <div
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  justifyContent: 'space-between',
                  gap: 16,
                  marginBottom: 22,
                }}
              >
                <div>
                  {running ? (
                    <React.Fragment>
                      <h1
                        style={{
                          fontSize: 24,
                          fontWeight: 600,
                          letterSpacing: '-0.02em',
                          margin: '0 0 6px',
                        }}
                      >
                        Analyzing batch…
                      </h1>
                      <p style={{ fontSize: 14, color: T.sub, margin: 0 }}>
                        Analyzing {Math.min(done + 1, files.length)} of {files.length}. Results
                        appear as each call finishes.
                      </p>
                    </React.Fragment>
                  ) : (
                    <React.Fragment>
                      <h1
                        style={{
                          fontSize: 24,
                          fontWeight: 600,
                          letterSpacing: '-0.02em',
                          margin: '0 0 4px',
                        }}
                      >
                        Results
                      </h1>
                      <p style={{ fontSize: 14, color: T.sub, margin: 0 }}>
                        {analyzed} call{analyzed !== 1 ? 's' : ''} analyzed
                        {summary.skipped > 0 && ' · ' + summary.skipped + ' not analyzed'}
                      </p>
                    </React.Fragment>
                  )}
                </div>
                {!running && (
                  <button
                    onClick={startOver}
                    style={{
                      background: T.surface,
                      color: T.ink,
                      border: '1px solid ' + T.border,
                      borderRadius: 10,
                      fontSize: 13.5,
                      fontWeight: 500,
                      padding: '10px 16px',
                      cursor: 'pointer',
                      flex: '0 0 auto',
                    }}
                  >
                    Start over
                  </button>
                )}
              </div>

              {running && (
                <div
                  style={{
                    height: 6,
                    borderRadius: 99,
                    background: T.border,
                    overflow: 'hidden',
                    marginBottom: 24,
                  }}
                >
                  <div
                    style={{
                      height: '100%',
                      background: T.red,
                      borderRadius: 99,
                      width: progressPct,
                      transition: 'width .4s ease',
                    }}
                  ></div>
                </div>
              )}

              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(2,1fr)',
                  gap: 12,
                  marginBottom: 28,
                }}
              >
                {[
                  {
                    label: 'Safe',
                    value: summary.safe,
                    dot: (
                      <span
                        style={{ width: 8, height: 8, borderRadius: 99, background: T.ink }}
                      ></span>
                    ),
                    color: T.ink,
                  },
                  {
                    label: 'Suspicious',
                    value: summary.suspicious,
                    dot: (
                      <span
                        style={{ width: 8, height: 8, borderRadius: 99, background: T.red }}
                      ></span>
                    ),
                    color: T.red,
                  },
                ].map((s, i) => (
                  <div
                    key={i}
                    style={{
                      border: '1px solid ' + T.border,
                      borderRadius: 12,
                      background: T.surface,
                      padding: '16px 18px',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 8 }}>
                      {s.dot}
                      <span style={{ fontSize: 12.5, color: T.sub, fontWeight: 500 }}>
                        {s.label}
                      </span>
                    </div>
                    <div
                      style={{
                        fontSize: 30,
                        fontWeight: 700,
                        letterSpacing: '-0.02em',
                        color: s.color,
                      }}
                    >
                      {s.value}
                    </div>
                  </div>
                ))}
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {results.map((r) => (
                  <ResultCard
                    key={r.uploadKey || r.id}
                    T={T}
                    r={r}
                    analysisView
                    open={!!expanded[r.id]}
                    onToggle={() => setExpanded((p) => ({ ...p, [r.id]: !p[r.id] }))}
                    tOpen={!!showT[r.id]}
                    onToggleT={() => setShowT((p) => ({ ...p, [r.id]: !p[r.id] }))}
                  />
                ))}
              </div>

              {running && (
                <div
                  style={{
                    border: '1px solid ' + T.border,
                    borderRadius: 14,
                    background: T.surface,
                    overflow: 'hidden',
                    marginTop: results.length ? 12 : 0,
                  }}
                >
                  {files
                    .filter((f) => f.status !== 'done')
                    .map((f) => (
                      <div
                        key={f.id}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: 12,
                          padding: '13px 16px',
                          borderBottom: '1px solid ' + T.soft,
                        }}
                      >
                        <span
                          style={{
                            flex: '0 0 20px',
                            width: 20,
                            height: 20,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                          }}
                        >
                          {f.status === 'queued' && (
                            <span
                              style={{
                                width: 9,
                                height: 9,
                                borderRadius: 99,
                                border: '1.5px solid ' + T.track,
                              }}
                            ></span>
                          )}
                          {f.status === 'analyzing' && (
                            <span
                              style={{
                                width: 16,
                                height: 16,
                                borderRadius: 99,
                                border: '2px solid ' + T.redBorder,
                                borderTopColor: T.red,
                                animation: 'spin .7s linear infinite',
                                display: 'inline-block',
                              }}
                            ></span>
                          )}
                        </span>
                        <div
                          style={{
                            flex: '1 1 auto',
                            minWidth: 0,
                            fontSize: 13.5,
                            fontWeight: 500,
                            whiteSpace: 'nowrap',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            color: f.status === 'queued' ? T.mut : T.ink,
                          }}
                        >
                          {f.name}
                        </div>
                        <span
                          style={{
                            fontSize: 12,
                            fontWeight: 500,
                            flex: '0 0 auto',
                            color: f.status === 'analyzing' ? T.red : T.mut,
                          }}
                        >
                          {f.status === 'analyzing' ? 'Analyzing' : 'Queued'}
                        </span>
                      </div>
                    ))}
                </div>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

function control(T, primary = false) {
  return {
    padding: '10px 14px',
    borderRadius: 9,
    border: '1px solid ' + (primary ? T.red : T.border),
    background: primary ? T.red : T.surface,
    color: primary ? '#fff' : T.ink,
    fontSize: 13,
  };
}
async function api(url, options) {
  const res = await fetch(url, options);
  const body = await res.json();
  if (!res.ok) throw new Error(body.error || 'Request failed');
  return body;
}
function Card({ T, r }) {
  const [open, setOpen] = useState(false),
    [transcript, setTranscript] = useState(false);
  return (
    <ResultCard
      T={T}
      r={r}
      open={open}
      onToggle={() => setOpen(!open)}
      tOpen={transcript}
      onToggleT={() => setTranscript(!transcript)}
    />
  );
}
function Notice({ T, children }) {
  return (
    <div
      role="status"
      style={{
        padding: '12px 16px',
        margin: '12px 0',
        border: '1px solid ' + T.border,
        borderRadius: 10,
        color: T.sub,
        fontSize: 13,
        lineHeight: 1.6,
      }}
    >
      {children}
    </div>
  );
}
function LibraryPane({ T, stats, revision, onAsk }) {
  const [category, setCategory] = useState(''),
    [status, setStatus] = useState(''),
    [page, setPage] = useState(1);
  const [data, setData] = useState({ recordings: [], total: 0 }),
    [loading, setLoading] = useState(true),
    [error, setError] = useState('');
  useEffect(() => {
    let active = true;
    setLoading(true);
    setError('');
    const params = new URLSearchParams({ page: String(page), page_size: '12' });
    if (category) params.set('category', category);
    if (status) params.set('status', status);
    api('/api/recordings?' + params)
      .then((d) => {
        if (active) setData(d);
      })
      .catch((e) => {
        if (active) setError(e.message);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [category, status, page, revision]);
  const changeCategory = (id) => {
    setCategory(category === String(id) ? '' : String(id));
    setPage(1);
  };
  return (
    <div className="page">
      <div className="toolbar" style={{ justifyContent: 'space-between', marginBottom: 26 }}>
        <div>
          <h1 style={{ margin: '0 0 8px', fontSize: 28 }}>Your recording library</h1>
          <p style={{ color: T.sub, margin: 0 }}>Every finding connected to its source.</p>
        </div>
        <button style={control(T, true)} onClick={() => onAsk('')}>
          Ask about your calls &rarr;
        </button>
      </div>
      <div className="toolbar" style={{ marginBottom: 22, fontSize: 13, color: T.sub }}>
        <span>
          <strong style={{ color: T.ink }}>{stats.total || 0}</strong> recordings
        </span>
        <span>&middot;</span>
        <span>{stats.searchable || 0} indexed for semantic search</span>
        <span>&middot;</span>
        <span>{stats.statuses?.suspicious || 0} suspicious</span>
      </div>
      <div className="category-grid" aria-label="Recording categories">
        {(stats.categories || []).map((c) => (
          <button
            key={c.id}
            onClick={() => changeCategory(c.id)}
            title={c.description}
            aria-pressed={category === String(c.id)}
            style={{
              textAlign: 'left',
              padding: 18,
              borderRadius: 12,
              border: '1px solid ' + (category === String(c.id) ? T.red : T.border),
              background: category === String(c.id) ? T.redSoft : T.surface,
              color: T.ink,
            }}
          >
            <div style={{ display: 'flex', gap: 8, justifyContent: 'space-between' }}>
              <span style={{ fontSize: 13, fontWeight: 600 }}>{c.name}</span>
              <strong style={{ fontSize: 22, color: T.red }}>{c.count}</strong>
            </div>
            <div style={{ fontSize: 11, color: T.sub, marginTop: 10 }}>
              {c.source === 'audio'
                ? 'Experimental audio cue'
                : c.fallback
                  ? 'Explicit conduct outside known categories'
                  : 'Evidence from the conversation'}
            </div>
          </button>
        ))}
      </div>
      <p style={{ fontSize: 12, color: T.mut }}>
        A call can appear in more than one category. Category counts are not additive.
      </p>
      <div className="toolbar" style={{ margin: '24px 0 18px' }}>
        <label style={{ fontSize: 13 }}>
          Status{' '}
          <select
            aria-label="Filter by status"
            value={status}
            onChange={(e) => {
              setStatus(e.target.value);
              setPage(1);
            }}
            style={control(T)}
          >
            <option value="">All recordings</option>
            <option value="suspicious">Suspicious</option>
            <option value="safe">Safe</option>
            <option value="skipped">Not analyzed</option>
          </select>
        </label>
        <button
          onClick={() => {
            setCategory('');
            setStatus('');
            setPage(1);
          }}
          style={control(T)}
        >
          Clear filters
        </button>
        <span style={{ marginLeft: 'auto', fontSize: 12, color: T.sub }}>
          {data.total} recording{data.total !== 1 ? 's' : ''}
        </span>
      </div>
      {error && <Notice T={T}>{error}</Notice>}
      {loading ? (
        <Notice T={T}>Loading library...</Notice>
      ) : (
        <div style={{ display: 'grid', gap: 14 }}>
          {data.recordings.map((r) => (
            <Card key={r.id} T={T} r={r} />
          ))}
          {!data.recordings.length && (
            <Notice T={T}>
              {stats.total
                ? 'No recordings match these filters.'
                : 'Your library is empty. Open Analyze to upload WAV files, or run python main.py to import the demo recordings.'}
            </Notice>
          )}
        </div>
      )}
      {data.total > 12 && (
        <div className="toolbar" style={{ justifyContent: 'center', marginTop: 24 }}>
          <button style={control(T)} disabled={page === 1} onClick={() => setPage(page - 1)}>
            Previous
          </button>
          <span>
            {page} / {Math.ceil(data.total / 12)}
          </span>
          <button
            style={control(T)}
            disabled={page * 12 >= data.total}
            onClick={() => setPage(page + 1)}
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
function SearchDetails({ T, response, stats }) {
  const d = response.diagnostics || {},
    p = d.plan || {},
    filters = d.filters || {
      status: p.status_filter,
      categories: p.category_filters || [],
      recording_ids: p.recording_ids || [],
    };
  const name = (id) => (stats.categories || []).find((c) => c.id === id)?.name || id;
  const scopeLabel = (s) =>
    ({
      inferred_categories: 'Automatically selected categories',
      explicit_categories: 'Explicit category filters',
      referenced_recordings: 'Previously referenced recordings',
      global: 'All eligible recordings',
      global_fallback: 'Broader library fallback',
    })[s] || s;
  return (
    <details style={{ margin: '16px 0', fontSize: 12, color: T.sub }}>
      <summary>How this search worked</summary>
      <div
        style={{
          padding: 16,
          marginTop: 10,
          background: T.surfaceAlt,
          borderRadius: 10,
          lineHeight: 1.8,
          overflowX: 'auto',
        }}
      >
        <div>
          <strong>Search query:</strong> {p.query}
        </div>
        <div>
          <strong>Selected categories:</strong>{' '}
          {(p.inferred_categories || []).map(name).join(', ') ||
            'None (broad query or explicit scope)'}
        </div>
        <div>
          <strong>Search scope:</strong>{' '}
          {d.scope
            ? scopeLabel(d.scope)
            : 'Earlier search; subject to explicit filters and referenced results'}
        </div>
        {d.strategy === 'category_first_evidence_fallback' && (
          <div>
            <strong>Broader fallback:</strong>{' '}
            {d.broadened
              ? d.fallback_reason === 'no_candidates'
                ? 'Used: selected categories had no retrieval candidates'
                : 'Used: no candidate in the selected categories passed evidence checking'
              : 'Not used'}
          </div>
        )}
        <div>
          <strong>Explicit filters:</strong>{' '}
          {[filters.status, ...filters.categories.map(name)].filter(Boolean).join(', ') || 'None'}
        </div>
        {!!filters.recording_ids.length && (
          <div>
            <strong>Follow-up scope:</strong> {filters.recording_ids.length} previously returned
            recording(s)
          </div>
        )}
        <div>
          <strong>Retrieval:</strong> keyword (BM25) + summary vectors &rarr; reciprocal rank fusion
        </div>
        <div>
          <strong>Reranking:</strong> {d.reranking || 'not run'} &middot; {d.candidate_count || 0}{' '}
          candidates &middot; {((d.elapsed_ms || 0) / 1000).toFixed(2)}s
        </div>
        {!!d.stages?.length && (
          <ol>
            {d.stages.map((s, i) => (
              <li key={i}>
                {scopeLabel(s.scope)}
                {s.categories?.length ? ': ' + s.categories.map(name).join(', ') : ''} &middot;{' '}
                {s.candidate_count} candidates &middot;{' '}
                {s.supported_count == null
                  ? 'matches not verified'
                  : s.supported_count + ' supported matches'}{' '}
                &middot;{' '}
                {Object.entries(s.branch_counts || {})
                  .map(([branch, count]) => branch + ': ' + count)
                  .join(' / ')}
              </li>
            ))}
          </ol>
        )}
        {d.broadened && (
          <div>
            The ranks below are from the broader search. Each stage fuses its own keyword and vector
            ranks.
          </div>
        )}
        <table
          style={{ width: '100%', marginTop: 12, textAlign: 'left', borderCollapse: 'collapse' }}
        >
          <thead>
            <tr>
              <th>Recording</th>
              <th>Before reranking</th>
              <th>Final rank</th>
              <th>Retrieval contributions</th>
            </tr>
          </thead>
          <tbody>
            {(d.candidates || []).map((c) => (
              <tr key={c.id}>
                <td style={{ padding: '6px 6px 6px 0' }}>{c.file}</td>
                <td>{c.hybrid_rank}</td>
                <td>
                  {response.results.findIndex((r) => r.id === c.id) >= 0
                    ? response.results.findIndex((r) => r.id === c.id) + 1
                    : 'Excluded'}
                </td>
                <td>{(c.contributions || []).map((x) => x.branch + ' #' + x.rank).join(' / ')}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p>
          Ranks and similarity scores describe retrieval, not confidence that a person is in danger.
        </p>
      </div>
    </details>
  );
}
function AskPane({ T, stats, draft }) {
  const [query, setQuery] = useState(''),
    [rerank, setRerank] = useState(true),
    [busy, setBusy] = useState(false),
    [error, setError] = useState('');
  const [messages, setMessages] = useState(() => {
    try {
      const saved = JSON.parse(sessionStorage.getItem('call-chat') || '[]');
      return Array.isArray(saved) ? saved : [];
    } catch {
      return [];
    }
  });
  useEffect(() => {
    if (draft.text) setQuery(draft.text);
  }, [draft]);
  useEffect(() => {
    try {
      sessionStorage.setItem('call-chat', JSON.stringify(messages.slice(-8)));
    } catch {}
  }, [messages]);
  const send = async (e, text) => {
    if (e) e.preventDefault();
    const question = (text || query).trim();
    if (!question || busy) return;
    setBusy(true);
    setError('');
    setQuery('');
    const prior = [...messages].reverse().find((m) => m.response);
    const history = messages
      .slice(-6)
      .map((m) => ({ role: m.role, content: m.text || m.response?.answer || '' }));
    const context = messages.slice(-7);
    setMessages([...context, { role: 'user', text: question }]);
    try {
      const response = await api('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: question,
          history,
          reference_ids: (prior?.response?.results || []).map((r) => r.id),
          rerank,
        }),
      });
      setMessages((prev) => [...prev, { role: 'assistant', response }]);
    } catch (e) {
      setError(e.message);
      setQuery(question);
    } finally {
      setBusy(false);
    }
  };
  const examples = [
    'A husband threatens his wife',
    'Private photos used to demand payment',
    'An angry customer without threats',
    'The midnight mailbox campaign',
  ];
  return (
    <div className="page search-layout">
      <div className="toolbar" style={{ justifyContent: 'space-between' }}>
        <div>
          <h1 style={{ fontSize: 28, margin: '0 0 8px' }}>Ask your recordings</h1>
          <p style={{ margin: 0, color: T.sub }}>
            Find the conversation. Read the evidence. Hear the source.
          </p>
        </div>
        <button
          style={control(T)}
          disabled={busy}
          onClick={() => {
            setMessages([]);
            setError('');
          }}
        >
          New conversation
        </button>
      </div>
      {!messages.length && (
        <div
          style={{
            margin: '28px 0',
            display: 'grid',
            gridTemplateColumns: 'repeat(2,minmax(0,1fr))',
            gap: 10,
          }}
        >
          {examples.map((x) => (
            <button
              key={x}
              disabled={busy}
              style={{ ...control(T), textAlign: 'left', lineHeight: 1.6 }}
              onClick={() => send(null, x)}
            >
              {x} &rarr;
            </button>
          ))}
        </div>
      )}
      <div aria-live="polite" style={{ display: 'grid', gap: 22, margin: '28px 0' }}>
        {messages.map((m, i) =>
          m.role === 'user' ? (
            <div
              key={i}
              style={{
                justifySelf: 'end',
                maxWidth: '85%',
                background: T.soft,
                padding: '12px 18px',
                borderRadius: '14px 14px 3px 14px',
                lineHeight: 1.6,
              }}
            >
              {m.text}
            </div>
          ) : (
            <div key={i}>
              <div
                style={{
                  fontSize: 11,
                  letterSpacing: 1.5,
                  fontWeight: 700,
                  color: T.red,
                  marginBottom: 10,
                }}
              >
                CALL INTELLIGENCE
              </div>
              {m.response.answer && (
                <p style={{ lineHeight: 1.75, whiteSpace: 'pre-line' }}>
                  {plainDashes(m.response.answer)}
                </p>
              )}
              {!!m.response.citations?.length && (
                <div className="toolbar" style={{ marginBottom: 14 }}>
                  {m.response.citations.map((c) => (
                    <a
                      key={c.recording_id}
                      href={'#result-' + i + '-' + c.recording_id}
                      style={{ fontSize: 12, color: T.sub }}
                    >
                      [{c.number}] {c.file}
                    </a>
                  ))}
                </div>
              )}
              {m.response.notices.map((n, j) => (
                <Notice key={j} T={T}>
                  {n}
                </Notice>
              ))}
              <div style={{ display: 'grid', gap: 14 }}>
                {m.response.results.map((r) => (
                  <div id={'result-' + i + '-' + r.id} key={r.id}>
                    <Card T={T} r={r} />
                  </div>
                ))}
              </div>
              <SearchDetails T={T} response={m.response} stats={stats} />
            </div>
          ),
        )}
      </div>
      {busy && <Notice T={T}>Searching recordings and matching evidence...</Notice>}
      {error && <Notice T={T}>{error}</Notice>}
      <form
        onSubmit={send}
        style={{
          border: '1px solid ' + T.border,
          background: T.surface,
          padding: 16,
          borderRadius: 14,
        }}
      >
        <textarea
          aria-label="Ask about recordings"
          placeholder="Find a call, or ask a follow-up question..."
          value={query}
          maxLength={2000}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              send(e);
            }
          }}
          rows={3}
          style={{
            resize: 'vertical',
            width: '100%',
            border: 'none',
            outline: 'none',
            background: 'transparent',
            color: T.ink,
            lineHeight: 1.6,
          }}
        />
        <div className="toolbar" style={{ justifyContent: 'space-between' }}>
          <label style={{ fontSize: 12, color: T.sub }}>
            <input
              type="checkbox"
              checked={rerank}
              onChange={(e) => setRerank(e.target.checked)}
              disabled={busy}
            />{' '}
            Verify evidence and rerank
          </label>
          <button disabled={busy || !query.trim()} style={control(T, true)} type="submit">
            {busy ? 'Searching...' : 'Search'}
          </button>
        </div>
      </form>
      <p style={{ fontSize: 11, color: T.mut, lineHeight: 1.6 }}>
        Answers refer to retrieved recordings. No matching evidence means no supported answer. Audio
        playback starts only when you press play.
      </p>
    </div>
  );
}
function App() {
  const [theme, setTheme] = useState(() =>
    localStorage.getItem('theme') === 'light' ? 'light' : 'dark',
  );
  const T = THEMES[theme];
  const [tab, setTab] = useState('analyze'),
    [revision, setRevision] = useState(0),
    [stats, setStats] = useState({}),
    [error, setError] = useState(''),
    [draft, setDraft] = useState({ text: '' });
  useEffect(() => {
    document.body.style.background = T.pageBg;
    document.body.style.color = T.ink;
    localStorage.setItem('theme', theme);
    document.documentElement.style.colorScheme = theme;
  }, [theme]);
  useEffect(() => {
    api('/api/categories')
      .then(setStats)
      .catch((e) => setError(e.message));
  }, [revision]);
  const onUpdated = () => setRevision((x) => x + 1);
  const onAsk = (text) => {
    setDraft({ text });
    setTab('ask');
  };
  return (
    <div style={{ minHeight: '100vh', background: T.pageBg, color: T.ink }}>
      <TopBar T={T} theme={theme} onToggle={() => setTheme(theme === 'light' ? 'dark' : 'light')} />
      <div style={{ borderBottom: '1px solid ' + T.border }}>
        <nav className="nav-row" aria-label="Main navigation">
          {[
            ['analyze', 'Analyze'],
            ['library', 'Library'],
            ['ask', 'Ask'],
          ].map(([id, name]) => (
            <button
              key={id}
              aria-current={tab === id ? 'page' : undefined}
              style={{ ...control(T, tab === id), minWidth: 90 }}
              onClick={() => setTab(id)}
            >
              {name}
            </button>
          ))}
        </nav>
      </div>
      {error && <Notice T={T}>{error}</Notice>}
      <div hidden={tab !== 'analyze'}>
        <AnalyzePane T={T} theme={theme} onUpdated={onUpdated} />
      </div>
      <div hidden={tab !== 'library'}>
        <LibraryPane T={T} stats={stats} revision={revision} onAsk={onAsk} />
      </div>
      <div hidden={tab !== 'ask'}>
        <AskPane T={T} stats={stats} draft={draft} />
      </div>
    </div>
  );
}
ReactDOM.createRoot(document.getElementById('root')).render(<App />);
