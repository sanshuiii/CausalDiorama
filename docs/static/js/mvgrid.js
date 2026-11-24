/* The multiverse as a grid: 7 subjects across, 4 backgrounds x 2 motions down, every world an edit of the real video.
   Tiles use 384-px thumbnail encodes with poster frames, load only when visible and play in sync. */
(function(){
  const VID='static/videos/';
  const C=window.CASES||{}; const M=C.multiverse; const host=document.getElementById('mv-grid'); if(!M||!host) return;
  const dims=M.dims, subjects=dims.find(d=>d.key==='subj').options, bgs=dims.find(d=>d.key==='bg').options, motions=dims.find(d=>d.key==='motion').options;
  const thumb=v=>VID+v.replace('multiverse_0282/','multiverse_0282/thumb/');
  const poster=v=>VID+v.replace('multiverse_0282/','multiverse_0282/poster/').replace(/\.mp4$/,'.jpg');
  let tiles=[], master=null, io=null;
  function build(){
    tiles.forEach(t=>{t.v.pause(); t.v.removeAttribute('src'); t.v.load();}); tiles=[]; if(io) io.disconnect();
    const set=M; const byKey={}; set.worlds.forEach(w=>byKey[w.bg+'|'+w.subj+'|'+w.motion]=w);
    const g=document.createElement('div'); g.className='mvgrid'; g.style.gridTemplateColumns='auto repeat(7, minmax(96px, 1fr))';
    g.appendChild(Object.assign(document.createElement('div'),{className:'mvcorner'}));
    subjects.forEach(([k,label])=>{const h=document.createElement('div');h.className='mvcol';h.textContent=label;g.appendChild(h);});
    bgs.forEach(([bk,blabel])=>motions.forEach(([mk,mlabel])=>{
      const r=document.createElement('div'); r.className='mvrow'; r.innerHTML='<b>'+blabel+'</b><span>'+(mk==='original'?'Original motion':'Swims in from the right')+'</span>'; g.appendChild(r);
      subjects.forEach(([sk])=>{ const w=byKey[bk+'|'+sk+'|'+mk]; const t=document.createElement('div'); t.className='mvtile'+(w&&w.source?' src':'');
        if(!w){ t.classList.add('empty'); g.appendChild(t); return; }
        const v=document.createElement('video'); v.muted=true; v.loop=true; v.playsInline=true; v.preload='none'; v.poster=poster(w.video); v.dataset.src=thumb(w.video); v.style.aspectRatio=set.aspect;
        t.appendChild(v); if(w.source){const s=document.createElement('span');s.className='mvsrc';s.textContent='Source';t.appendChild(s);}
        g.appendChild(t); tiles.push({v,w,t});
      });
    }));
    host.replaceChildren(g);
    io=new IntersectionObserver(es=>es.forEach(e=>{ const v=e.target; if(e.isIntersecting){ if(!v.getAttribute('src')){ v.src=v.dataset.src; } v.play().catch(()=>{}); if(!master||master.paused) master=v; } else { v.pause(); } }),{rootMargin:'120px'});
    tiles.forEach(t=>io.observe(t.v));
  }
  /* keep the visible tiles on one clock (all clips of a side have the same length) */
  setInterval(()=>{ if(!master) return; const t0=master.currentTime; tiles.forEach(t=>{ const v=t.v; if(v!==master && !v.paused && v.readyState>=2 && Math.abs(v.currentTime-t0)>0.12) v.currentTime=t0; }); },700);
  build();
})();
