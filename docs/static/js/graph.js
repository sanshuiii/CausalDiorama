/* Interactive concept graph of the basketball case. The three states are SVGs drawn with the paper's Fig. 3 code
   (static/images/graph/state_{source,do,nodep}.svg, same geometry). Choosing an intervention replays it as an
   animation: a do(·) arrow enters from the left, the intervened node and its lines blink and turn into the new
   ones, then the change propagates left to right; the video plays meanwhile. */
(function(){
  const GRAPH='static/images/graph/', VID='static/videos/method_basketball/', SVGNS='http://www.w3.org/2000/svg';
  const STATES=[
    { key:'source', btn:'Source', label:'Source', video:'source.mp4' },
    { key:'do', btn:'Intervention', label:'Counterfactual', video:'ours_shot.mp4' },
    { key:'nodep', btn:'Without dependencies', label:'Without dependencies', video:'nocausal.mp4' }
  ];
  const INSTR='Player in red <b>shoots directly</b> and scores.';
  const host=document.getElementById('m-graph'); if(!host) return;
  const ch=document.getElementById('m-choices'), out=document.getElementById('m-result'), ins=document.getElementById('m-instruction');
  ins.innerHTML=INSTR;
  const docs={}; let run=0;
  const sleep=ms=>new Promise(r=>setTimeout(r,ms));

  /* tag the movable units of one state layer: boxes (b0..b3), participation paths (p0..p5), chain lines (c0..c2), new edge (ne) */
  function units(svg,layer){
    const list=[];
    svg.querySelectorAll('g[id^="'+layer+'-"]').forEach(g=>{
      const id=g.id.slice(layer.length+1); let m;
      if(id==='participation'){ [...g.children].forEach((p,i)=>list.push([p,'p'+i])); }
      else if(id==='chain'){ [...g.children].forEach((l,i)=>list.push([l,'c'+i])); }
      else if((m=id.match(/^box(\d)/))){ list.push([g,'b'+m[1]]); }
      else if(id==='new-edge'){ list.push([g,'ne']); }
    });
    list.forEach(([el,k])=>{ el.dataset.layer=layer; el.dataset.k=k; });
    return list;
  }
  /* an svg holding the source layer and, for a target state, its after layer (hidden) */
  function build(target){
    const svg=docs.source.cloneNode(true); units(svg,'before');
    if(target){
      const aft=docs[target].cloneNode(true); units(aft,'after');
      aft.querySelectorAll('[data-layer="after"]').forEach(el=>{ const c=document.importNode(el,true); c.classList.add('hid'); svg.appendChild(c); });
    }
    svg.removeAttribute('width'); svg.removeAttribute('height'); svg.classList.add('cgraph');
    return svg;
  }
  const q=(svg,layer,k)=>svg.querySelector('[data-layer="'+layer+'"][data-k="'+k+'"]');
  function swap(svg,keys){ keys.forEach(k=>{ const a=q(svg,'before',k), b=q(svg,'after',k); if(a) a.classList.add('hid'); if(b) b.classList.remove('hid'); }); }
  async function blink(els,token){ els.forEach(e=>e&&e.classList.add('blink')); await sleep(600); els.forEach(e=>e&&e.classList.remove('blink')); return token===run; }
  /* two blinks: the first shows the old unit out and in, the second takes the old one out and brings the new one in */
  async function turn(svg,keys,token,extra){
    const olds=keys.map(k=>q(svg,'before',k)).concat(extra||[]), news=keys.map(k=>q(svg,'after',k));
    if(!await blink(olds,token)) return false;
    olds.forEach(e=>e&&e.classList.add('out')); await sleep(320); if(token!==run) return false;
    olds.forEach(e=>{ if(!e) return; e.classList.remove('out'); if(e.dataset.layer==='before') e.classList.add('hid'); else e.remove(); });
    news.forEach(e=>{ if(!e) return; e.classList.remove('hid'); e.classList.add('in'); });
    await sleep(320); news.forEach(e=>e&&e.classList.remove('in')); return token===run;
  }
  function doArrow(svg){
    const g=document.createElementNS(SVGNS,'g'); g.setAttribute('class','doarrow hid');
    const l=document.createElementNS(SVGNS,'line'); l.setAttribute('x1','-8'); l.setAttribute('y1','64'); l.setAttribute('x2','15.5'); l.setAttribute('y2','64');
    l.setAttribute('stroke','#d95f02'); l.setAttribute('stroke-width','1'); l.setAttribute('marker-end','url(#ah_d95f02)'); g.appendChild(l);
    const t=document.createElementNS(SVGNS,'text'); t.setAttribute('x','7.8'); t.setAttribute('y','59.5'); t.setAttribute('font-size','6.2'); t.setAttribute('text-anchor','middle');
    t.setAttribute('font-weight','bold'); t.setAttribute('font-style','italic'); t.setAttribute('fill','#d95f02'); t.textContent='do(·)'; g.appendChild(t);
    svg.appendChild(g); return g;
  }
  function cell(label,video,ours,wait){
    const el=document.createElement('div'); el.className='cell'+(ours?' ours':'')+(label==='Source'?' source':'');
    const v=document.createElement('video'); v.muted=true; v.loop=true; v.playsInline=true; v.preload='auto'; v.src=VID+video; el.appendChild(v);
    if(wait){ const o=document.createElement('div'); o.className='loading'; o.innerHTML='<div class="spinner"></div>'; el.appendChild(o); }
    else v.autoplay=true;
    return el;
  }
  function setButtons(i){ ch.querySelectorAll('.choice').forEach((b,j)=>b.classList.toggle('active',i===j)); ins.classList.toggle('off',i===0); }

  function showStatic(i){ run++; const s=STATES[i]; setButtons(i);
    const svg=build(i?s.key:null); if(i){ const all=[...svg.querySelectorAll('[data-layer="after"]')].map(e=>e.dataset.k); swap(svg,all); }
    host.replaceChildren(svg); out.replaceChildren(cell(s.label,s.video,i===1,false)); }

  async function animate(i){ const token=++run; const s=STATES[i]; setButtons(i);
    const svg=build(s.key); host.replaceChildren(svg);
    out.replaceChildren(cell(s.label,s.video,i===1,false));                       /* the video plays right away */
    const B=k=>q(svg,'before',k), arrow=doArrow(svg);
    await sleep(350); if(token!==run) return;
    arrow.classList.remove('hid'); arrow.classList.add('slide'); await sleep(420); arrow.classList.remove('slide'); if(token!==run) return;   /* do(·) slides in from the left */
    if(!await turn(svg,['b0','p3','p4','c0'],token,[arrow])) return;                 /* the pass turns into red shoots; the arrow leaves with it */
    await sleep(200); if(token!==run) return;
    const steps = s.key==='do' ? [['b1','p0','p1','p2','c1'],['b2','c2']] : [];     /* propagation, left to right */
    for(const keys of steps){ if(!await turn(svg,keys,token)) return; await sleep(200); if(token!==run) return; }
    if(s.key==='do'){ if(!await turn(svg,['b3','p5','ne'],token)) return; }         /* the score gets its new dependency */
    const all=[...svg.querySelectorAll('[data-layer="after"]')].map(e=>e.dataset.k); swap(svg,all);
  }
  Promise.all(STATES.map(s=>fetch(GRAPH+'state_'+s.key+'.svg').then(r=>r.text()).then(t=>{docs[s.key]=new DOMParser().parseFromString(t,'image/svg+xml').documentElement;}))).then(()=>showStatic(0));
  STATES.forEach((s,i)=>{const b=document.createElement('button');b.className='choice';b.textContent=s.btn;b.addEventListener('click',()=>i?animate(i):showStatic(0));ch.appendChild(b);});
})();
