import { useEffect, useState } from 'react'
import LabelStudio from './LabelStudio.jsx'

const tabs = ['Projects', 'Datasets', 'Label', 'Train', 'Models']

async function api(path, opts) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts
  })
  return res.json()
}

export default function App() {
  const [tab, setTab] = useState('Projects')
  const [health, setHealth] = useState(null)
  const [projects, setProjects] = useState([])
  const [datasets, setDatasets] = useState([])
  const [runs, setRuns] = useState([])
  const [evals, setEvals] = useState({})
  const [name, setName] = useState('Parking lot demo')
  const [epochs, setEpochs] = useState(20)
  const [dsId, setDsId] = useState('')
  const [recipe, setRecipe] = useState('v2-photo')
  const [previews, setPreviews] = useState([])
  const [fmt, setFmt] = useState('npz')
  const [backend, setBackend] = useState('auto')
  const [augment, setAugment] = useState('v1-none')
  const [task, setTask] = useState('auto')
  const [detail, setDetail] = useState(null)
  const [preds, setPreds] = useState(null)
  const [exports, setExports] = useState([])
  const [compare, setCompare] = useState([])
  const [order, setOrder] = useState(null)
  const [versions, setVersions] = useState([])
  const [msg, setMsg] = useState('')

  const refresh = async () => {
    setHealth(await api('/api/health'))
    setProjects((await api('/api/projects')).projects || [])
    const ds = (await api('/api/datasets')).datasets || []
    setDatasets(ds)
    if (!dsId && ds[0]) setDsId(ds[0].id)
    const r = (await api('/api/runs')).runs || []
    setRuns(r)
    const ev = {}
    for (const run of r.slice(0, 5)) {
      try {
        ev[run.id] = (await api(`/api/runs/${run.id}/eval`)).eval
      } catch { ev[run.id] = null }
    }
    setEvals(ev)
    try {
      setCompare((await api('/api/runs/compare')).runs || [])
    } catch { setCompare([]) }
    try {
      setExports((await api('/api/exports')).exports || [])
    } catch { setExports([]) }
  }

  const openDetail = async (runId) => {
    setDetail(await api(`/api/runs/${runId}`))
    setPreds(await api(`/api/runs/${runId}/predictions`))
  }

  useEffect(() => { refresh().catch(() => {}) }, [tab])

  const upload = async (e) => {
    const file = e.target.files[0]
    if (!file || !dsId) return
    const form = new FormData()
    form.append('file', file)
    await fetch(`/api/datasets/${dsId}/images`, { method: 'POST', body: form })
    setMsg(`uploaded ${file.name}`)
    refresh()
  }

  return (
    <div className="shell">
      <aside className="side">
        <div className="brand">VisionForge<small>local-first Mac trainer</small></div>
        <nav className="nav">
          {tabs.map((t) => (
            <button key={t} className={t === tab ? 'active' : ''} onClick={() => setTab(t)}>{t}</button>
          ))}
        </nav>
        <div className="health">device: {health ? health.device : '…'}</div>
      </aside>
      <main className="main">
        <h1>{tab}</h1>
        <p className="muted">Everything runs on this Mac. No uploads, no cloud queue. {msg}</p>

        {tab === 'Projects' && (
          <div className="panel">
            <div className="row">
              <input value={name} onChange={(e) => setName(e.target.value)} />
              <button className="primary" onClick={async () => {
                await api('/api/projects', { method: 'POST', body: JSON.stringify({ name }) })
                refresh()
              }}>New project</button>
            </div>
            <table><tbody>
              {projects.map((p) => <tr key={p.id}><td>{p.name}</td><td>{p.task}</td><td>{p.id}</td></tr>)}
            </tbody></table>
          </div>
        )}

        {tab === 'Datasets' && (
          <>
            <div className="panel">
              <div className="row">
                <select value={dsId} onChange={(e) => setDsId(e.target.value)}>
                  {datasets.map((d) => <option key={d.id} value={d.id}>{d.id}</option>)}
                </select>
                <label className="file-btn">Upload image
                  <input type="file" accept=".png,.jpg,.jpeg" onChange={upload} hidden />
                </label>
              </div>
              <table><tbody>
                {datasets.map((d) => <tr key={d.id}><td>{d.id}</td><td>v{d.version}</td><td>{d.project_id}</td></tr>)}
              </tbody></table>
              <div className="row">
                <button onClick={async () => {
                  if (!dsId) return
                  setVersions(((await api(`/api/datasets/${dsId}/versions`)).versions) || [])
                }}>Show versions</button>
                <span className="muted">{versions.map((v) => `v${v.version}`).join(', ')}</span>
              </div>
              <p className="muted">Video import runs from CLI: python visionforge.py import-video --file clip.mp4 --fps 1</p>
            </div>
            <div className="panel">
              <h2>Augmentation preview</h2>
              <div className="row">
                {['v1-none', 'v1-flip', 'v2-photo', 'v3-rotate'].map((r) => (
                  <button key={r} className={r === recipe ? 'primary' : ''} onClick={() => setRecipe(r)}>{r}</button>
                ))}
                <button className="primary" onClick={async () => {
                  if (!dsId) return
                  const res = await api(`/api/datasets/${dsId}/augment-preview`, {
                    method: 'POST', body: JSON.stringify({ recipe, n: 4 })
                  })
                  setPreviews(res.previews || [])
                }}>Preview</button>
              </div>
              <div className="thumbs">
                {previews.map((p, i) => (
                  <figure key={i}><img src={p.data_url} alt={p.image} /><figcaption>{p.image}</figcaption></figure>
                ))}
              </div>
            </div>
          </>
        )}

        {tab === 'Label' && (
          <div className="panel">
            <div className="row">
              <select value={dsId} onChange={(e) => { setDsId(e.target.value); setOrder(null) }}>
                {datasets.map((d) => <option key={d.id} value={d.id}>{d.id}</option>)}
              </select>
              <button onClick={async () => {
                if (!dsId) return
                const res = await api(`/api/datasets/${dsId}/queue?n=50`)
                setOrder((res.queue || []).map((q) => q.image))
                setMsg(`${(res.queue || []).length} hardest-first`)
              }}>Hardest-first queue</button>
              {order && <button onClick={() => setOrder(null)}>Clear order</button>}
            </div>
            <LabelStudio datasetId={dsId} order={order} />
          </div>
        )}

        {tab === 'Train' && (
          <div className="panel">
            <div className="row">
              <select value={dsId} onChange={(e) => setDsId(e.target.value)}>
                {(datasets.length ? datasets : [{ id: '' }]).map((d) => (
                  <option key={d.id} value={d.id}>{d.id}</option>
                ))}
              </select>
              <input type="number" min="1" max="200" value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} />
              <select value={backend} onChange={(e) => setBackend(e.target.value)}>
                <option value="auto">auto backend</option>
                <option value="torch">torch / MPS</option>
                <option value="numpy">numpy</option>
              </select>
              <select value={augment} onChange={(e) => setAugment(e.target.value)}>
                <option value="v1-none">no augment</option>
                <option value="v1-flip">v1-flip</option>
                <option value="v2-photo">v2-photo</option>
                <option value="v3-rotate">v3-rotate</option>
              </select>
              <select value={task} onChange={(e) => setTask(e.target.value)}>
                <option value="auto">auto task</option>
                <option value="detect">detect</option>
                <option value="classify">classify</option>
              </select>
              <button className="primary" onClick={async () => {
                if (!dsId) return
                await api('/api/runs', { method: 'POST', body: JSON.stringify({ dataset_id: dsId, epochs, backend, augment, task }) })
                setTimeout(refresh, 1500)
              }}>Start run</button>
              <button onClick={refresh}>Refresh</button>
            </div>
            {compare.length > 0 && (
              <table className="matrix"><tbody>
                <tr><th>run</th><th>backend</th><th>augment</th><th>epochs</th><th>loss</th><th>acc</th></tr>
                {compare.slice(0, 8).map((c) => (
                  <tr key={c.id}><td>{c.id}</td><td>{c.backend || '—'}</td><td>{c.augment}</td><td>{c.epochs ?? '—'}</td><td>{c.final_loss ?? '—'}</td><td>{c.accuracy ?? '—'}</td></tr>
                ))}
              </tbody></table>
            )}
            {runs.slice(0, 5).map((r) => (
              <div key={r.id} className="run-block">
                <p className="muted">{r.id} — {r.status} <button onClick={() => openDetail(r.id)}>Details</button></p>
                <div className="chart">
                  {(r.metrics || []).map((m, i) => (
                    <span key={i} title={`ep ${m.epoch}: loss ${m.loss}`} style={{ height: `${Math.max(6, Math.min(120, (1.6 - m.loss) * 70))}px` }} />
                  ))}
                </div>
                {evals[r.id] && evals[r.id].mAP50 !== undefined && (
                  <p className="muted">mAP@0.5: <strong>{evals[r.id].mAP50}</strong> — mean IoU: <strong>{evals[r.id].mean_iou}</strong> — class acc: {evals[r.id].accuracy}</p>
                )}
                {evals[r.id] && evals[r.id].confusion_matrix && (
                  <table className="matrix"><tbody>
                    <tr><th>accuracy</th><td colSpan={evals[r.id].classes.length}>{evals[r.id].accuracy}</td></tr>
                    <tr><th>true ↓ / pred →</th>{evals[r.id].classes.map((c) => <th key={c}>{c}</th>)}</tr>
                    {evals[r.id].confusion_matrix.map((row, i) => (
                      <tr key={i}><th>{evals[r.id].classes[i]}</th>{row.map((v, j) => <td key={j}>{v}</td>)}</tr>
                    ))}
                  </tbody></table>
                )}
              </div>
            ))}
            {detail && (
              <div className="panel detail">
                <h2>Run {detail.run.id} <button onClick={() => { setDetail(null); setPreds(null) }}>Close</button></h2>
                <p className="muted">
                  task: {detail.config?.task || detail.config?.mode} — backend: {detail.config?.mode} —
                  train/val: {detail.config?.n_train}/{detail.config?.n_val} —
                  dataset v{detail.config?.dataset_version} ({detail.config?.manifest_sha})
                </p>
                <h2>Artifacts</h2>
                <table><tbody>
                  {detail.artifacts.map((a) => <tr key={a.name}><td>{a.name}</td><td>{a.bytes} bytes</td></tr>)}
                </tbody></table>
                {preds && preds.predictions && (
                  <>
                    <h2>Val predictions</h2>
                    <table className="matrix"><tbody>
                      <tr><th>image</th><th>true</th><th>pred</th><th>IoU</th></tr>
                      {preds.predictions.slice(0, 8).map((p, i) => (
                        <tr key={i}><td>{p.image}</td><td>{p.true_label}</td><td>{p.pred_label}</td><td>{p.iou}</td></tr>
                      ))}
                    </tbody></table>
                  </>
                )}
                <h2>Log</h2>
                <pre className="log">{detail.log.join('\n')}</pre>
              </div>
            )}
          </div>
        )}

        {tab === 'Models' && (
          <div className="panel">
            <h2>Registry + export</h2>
            <div className="row">
              <select value={fmt} onChange={(e) => setFmt(e.target.value)}>
                <option value="npz">npz weights</option>
                <option value="coreml">Core ML</option>
                <option value="onnx">ONNX</option>
              </select>
            </div>
            <table><tbody>
              {runs.map((r) => (
                <tr key={r.id}><td>{r.id}</td><td>{r.status}</td>
                  <td><button onClick={async () => {
                    const res = await api(`/api/runs/${r.id}/export`, {
                      method: 'POST', body: JSON.stringify({ format: fmt })
                    })
                    setMsg(res.artifact ? `exported: ${res.artifact}` : `stub: ${res.reason}`)
                    refresh()
                  }}>Export {fmt}</button></td></tr>
              ))}
            </tbody></table>
            <h2>Export registry</h2>
            <table><tbody>
              {exports.map((e) => (
                <tr key={e.id}><td>{e.format}</td><td>{e.run_id}</td><td>{e.status}</td>
                  <td>{e.artifact ? <a href={`/api/exports/${e.id}/download`}>download</a> : '—'}</td></tr>
              ))}
            </tbody></table>
          </div>
        )}
      </main>
    </div>
  )
}
