import { useEffect, useRef, useState } from 'react'

async function api(path, opts) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts
  })
  return res.json()
}

// Interactive box annotator. Keys: n/p next/prev image, s save, d delete last.
export default function LabelStudio({ datasetId, order }) {
  const [items, setItems] = useState([])
  const [images, setImages] = useState([])
  const [idx, setIdx] = useState(0)
  const [label, setLabel] = useState('red-circle')
  const [dirty, setDirty] = useState(false)
  const [msg, setMsg] = useState('')
  const canvasRef = useRef(null)
  const imgRef = useRef(null)
  const drawRef = useRef(null)

  useEffect(() => {
    if (!datasetId) return
    api(`/api/datasets/${datasetId}/manifest`).then((m) => {
      setItems(m.items || [])
      setImages(m.images || [])
      setIdx(0)
      setDirty(false)
    }).catch(() => {})
  }, [datasetId])

  const ordered = order && order.length
    ? [...order.filter((n) => images.includes(n)),
       ...images.filter((n) => !order.includes(n))]
    : images
  const imagesShown = ordered
  const name = imagesShown[idx % Math.max(imagesShown.length, 1)]
  const boxesOf = (imageName) =>
    (items.find((it) => it.image === imageName)?.boxes) || []

  const draw = () => {
    const canvas = canvasRef.current
    const img = imgRef.current
    if (!canvas || !img || !img.complete || !img.naturalWidth) return
    canvas.width = img.naturalWidth
    canvas.height = img.naturalHeight
    const ctx = canvas.getContext('2d')
    ctx.drawImage(img, 0, 0)
    ctx.lineWidth = 2
    for (const b of boxesOf(name)) {
      ctx.strokeStyle = '#1d6b45'
      ctx.strokeRect(b.x, b.y, b.w, b.h)
      ctx.fillStyle = '#1d6b45'
      ctx.font = '12px sans-serif'
      ctx.fillText(b.label || '', b.x + 3, b.y + 13)
    }
    const d = drawRef.current
    if (d) {
      ctx.strokeStyle = '#b3541e'
      ctx.strokeRect(d.x, d.y, d.w, d.h)
    }
  }

  useEffect(() => { draw() })

  const pos = (e) => {
    const r = canvasRef.current.getBoundingClientRect()
    const sx = canvasRef.current.width / r.width
    const sy = canvasRef.current.height / r.height
    return { x: (e.clientX - r.left) * sx, y: (e.clientY - r.top) * sy }
  }

  const onDown = (e) => {
    const p = pos(e)
    drawRef.current = { x: p.x, y: p.y, w: 0, h: 0 }
  }
  const onMove = (e) => {
    const d = drawRef.current
    if (!d) return
    const p = pos(e)
    d.w = p.x - d.x
    d.h = p.y - d.y
    draw()
  }
  const onUp = () => {
    const d = drawRef.current
    drawRef.current = null
    if (!d || Math.abs(d.w) < 5 || Math.abs(d.h) < 5) { draw(); return }
    const box = {
      x: Math.round(Math.min(d.x, d.x + d.w)),
      y: Math.round(Math.min(d.y, d.y + d.h)),
      w: Math.round(Math.abs(d.w)),
      h: Math.round(Math.abs(d.h)),
      label
    }
    setItems((prev) => {
      const next = prev.filter((it) => it.image !== name)
      next.push({ image: name, label, boxes: [...boxesOf(name), box] })
      return next
    })
    setDirty(true)
    draw()
  }

  const save = async () => {
    await api(`/api/datasets/${datasetId}/annotations`, {
      method: 'PUT',
      body: JSON.stringify({ items })
    })
    setDirty(false)
    setMsg(`saved ${items.length} items`)
  }

  const step = (dir) => {
    setIdx((i) => (i + dir + imagesShown.length) % Math.max(imagesShown.length, 1))
    setMsg('')
  }

  useEffect(() => {
    const onKey = (e) => {
      if (e.target.tagName === 'INPUT') return
      if (e.key === 'n') step(1)
      if (e.key === 'p') step(-1)
      if (e.key === 's') save()
      if (e.key === 'd') {
        setItems((prev) => prev.map((it) =>
          it.image === name ? { ...it, boxes: (it.boxes || []).slice(0, -1) } : it))
        setDirty(true)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  if (!datasetId) return <p className="muted">Pick a dataset to label.</p>
  if (!imagesShown.length) return <p className="muted">No images in this dataset yet.</p>

  return (
    <div>
      <div className="row">
        <button onClick={() => step(-1)}>← prev (p)</button>
        <strong>{name}</strong>
        <span className="muted">{(idx % Math.max(imagesShown.length, 1)) + 1} / {imagesShown.length}</span>
        <button onClick={() => step(1)}>next (n) →</button>
        <input value={label} onChange={(e) => setLabel(e.target.value)} style={{ minWidth: 140 }} />
        <button onClick={async () => {
          const res = await api(`/api/datasets/${datasetId}/propose`, {
            method: 'POST', body: JSON.stringify({ images: [name] })
          })
          const found = (res.proposals || {})[name] || []
          const mapped = found.map((b) => ({ ...b, label }))
          setItems((prev) => {
            const next = prev.filter((it) => it.image !== name)
            next.push({ image: name, label, boxes: mapped })
            return next
          })
          setDirty(true)
          setMsg(`${mapped.length} proposed`)
        }}>Propose</button>
        <button className="primary" onClick={save}>Save (s){dirty ? ' •' : ''}</button>
        <span className="muted">{msg}</span>
      </div>
      <div className="canvas-wrap">
        <img
          ref={imgRef}
          src={`/api/datasets/${datasetId}/images/${name}`}
          onLoad={draw}
          alt={name}
          style={{ display: 'none' }}
        />
        <canvas
          ref={canvasRef}
          onMouseDown={onDown}
          onMouseMove={onMove}
          onMouseUp={onUp}
        />
      </div>
      <p className="muted">Drag to draw a box. d deletes the last box. {boxesOf(name).length} boxes on this image.</p>
    </div>
  )
}
