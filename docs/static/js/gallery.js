/* Galleries: coverflow-style carousels. In a comparison gallery The active clip sits large in the centre
   with the source video and our result overlaid and a diagonal divider whose handle reveals either side; the previous
   and next clips show smaller at the sides, cut by the stage edges and faded. Chevrons, keyboard, and a horizontal
   drag or swipe (pointer events, so touch too) move between clips; the list wraps around. */
function mountGallery(hostId,set){
  const VID='static/videos/';
  const host=document.getElementById(hostId); const items=set&&set.items; if(!host||!items||!items.length) return;
  const compare=!!set.compare;
  const n=items.length, LEAN=0.09;
  let cur=0, slides=[];
  const stage=document.createElement('div'); stage.className='gstage';
  const prev=document.createElement('button'); prev.className='gchev l'; prev.setAttribute('aria-label','previous');
  prev.innerHTML='<svg viewBox="0 0 24 40"><path d="M20 3 L4 20 L20 37" fill="none" stroke="currentColor" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  const next=document.createElement('button'); next.className='gchev r'; next.setAttribute('aria-label','next');
  next.innerHTML='<svg viewBox="0 0 24 40"><path d="M4 3 L20 20 L4 37" fill="none" stroke="currentColor" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  const caption=document.createElement('div'); caption.className='input gcap';
  const dots=document.createElement('div'); dots.className='gdots';
  items.forEach((it,i)=>{
    const s=document.createElement('div'); s.className='gslide';
    const cmp=document.createElement('div'); cmp.className='gcmp'; cmp.style.aspectRatio=it.aspect;
    const mk=src=>{const v=document.createElement('video'); v.muted=true; v.loop=true; v.playsInline=true; v.preload='metadata'; v.src=VID+src; return v;};
    const vo=mk(it.ours), vs=compare?mk(it.source):null; if(vs) vs.className='gsrc';
    const line=document.createElement('div'); line.className='gline';
    const handle=document.createElement('div'); handle.className='ghandle'; handle.innerHTML='<span>&#8596;</span>';
    const l1=document.createElement('span'); l1.className='glabel l'; l1.textContent='Source';
    const l2=document.createElement('span'); l2.className='glabel r'; l2.textContent='Ours';
    cmp.appendChild(vo); if(compare) cmp.append(vs,line,handle,l1,l2); s.appendChild(cmp); stage.appendChild(s);
    const sl={el:s,cmp,vo,vs,line,handle,p:0.5,text:it.text,aspect:it.aspect};
    function setP(p){ if(!compare) return; sl.p=Math.max(0.04,Math.min(0.96,p)); const t=sl.p+LEAN, b=sl.p-LEAN;
      vs.style.clipPath=`polygon(0 0, ${t*100}% 0, ${b*100}% 100%, 0 100%)`;
      line.style.left=((t+b)/2*100)+'%'; line.style.transform=`translateX(-50%) rotate(${Math.atan2((t-b)*cmp.clientWidth, cmp.clientHeight)}rad)`;
      handle.style.left=(sl.p*100)+'%'; }
    sl.setP=setP; setP(0.5);
    handle.addEventListener('pointerdown',e=>{ if(i!==cur) return; e.preventDefault(); e.stopPropagation(); try{ handle.setPointerCapture(e.pointerId); }catch(_){}
      const r=cmp.getBoundingClientRect(); const mv=ev=>setP((ev.clientX-r.left)/r.width);
      handle.addEventListener('pointermove',mv); handle.addEventListener('pointerup',()=>handle.removeEventListener('pointermove',mv),{once:true}); });
    if(compare) vo.addEventListener('timeupdate',()=>{ if(Math.abs(vs.currentTime-vo.currentTime)>0.08) vs.currentTime=vo.currentTime; });
    s.addEventListener('click',()=>{ if(i!==cur && !dragMoved) go(i); });
    slides.push(sl);
  });
  const fadeL=document.createElement('div'); fadeL.className='gfade l'; const fadeR=document.createElement('div'); fadeR.className='gfade r';
  stage.append(fadeL,fadeR,prev,next);
  host.append(stage,caption,dots);
  items.forEach((_,i)=>{const d=document.createElement('button');d.className='gdot';d.setAttribute('aria-label','clip '+(i+1));d.addEventListener('click',()=>go(i));dots.appendChild(d);});
  /* layout: offset -1, 0, +1 are the visible slots; the rest wait hidden on their side */
  const SIDE=0.8, GAP=0.02, cw=()=>matchMedia('(max-width:700px)').matches?0.78:0.62;   /* the active slide's share of the stage width (see .gslide) */
  function layout(dx){
    const W=stage.clientWidth; dx=dx||0;
    slides.forEach((sl,i)=>{ let d=((i-cur)%n+n)%n; if(d>n/2) d-=n;
      const el=sl.el; el.classList.toggle('active',d===0);
      const CW=cw(); const sideShift=(CW/2+GAP+CW*SIDE/2)*W;          /* centre-to-centre distance between the active slide and a side slide */
      let x, sc, op, vis;
      if(d===0){ x=dx; sc=1; op=1; vis=true; }
      else if(Math.abs(d)===1){ x=Math.sign(d)*sideShift+dx; sc=SIDE; op=0.55; vis=true; }
      else { x=Math.sign(d)*(sideShift*1.9)+dx; sc=SIDE*0.9; op=0; vis=false; }
      el.style.transform=`translate(calc(-50% + ${x}px), -50%) scale(${sc})`; el.style.opacity=op; el.style.visibility=vis?'visible':'hidden'; el.style.zIndex=d===0?3:(Math.abs(d)===1?2:1);
    });
  }
  function go(i){ cur=((i%n)+n)%n; const [aw,ah]=slides[cur].aspect.split('/').map(Number); stage.style.aspectRatio=String((aw/ah)/cw()/1.06);   /* the active slide fills the stage height, with a small margin */
    slides.forEach((sl,j)=>{ const vids=[sl.vo,sl.vs].filter(Boolean); if(j===cur) vids.forEach(v=>v.play().catch(()=>{})); else vids.forEach(v=>v.pause()); });
    caption.innerHTML=slides[cur].text; dots.querySelectorAll('.gdot').forEach((d,j)=>d.classList.toggle('active',j===cur)); layout(0); }
  prev.addEventListener('click',e=>{e.stopPropagation();go(cur-1);}); next.addEventListener('click',e=>{e.stopPropagation();go(cur+1);});
  /* drag / swipe on the stage, except on the divider handle */
  let drag=null, dragMoved=false;
  stage.addEventListener('pointerdown',e=>{ if(e.target.closest('.ghandle')||e.target.closest('.gchev')) return; drag={x:e.clientX,y:e.clientY,id:e.pointerId,moved:false}; dragMoved=false; });
  stage.addEventListener('pointermove',e=>{ if(!drag||e.pointerId!==drag.id) return; const dx=e.clientX-drag.x, dy=e.clientY-drag.y;
    if(!drag.moved && Math.abs(dx)>8 && Math.abs(dx)>Math.abs(dy)){ drag.moved=true; dragMoved=true; try{ stage.setPointerCapture(e.pointerId); }catch(_){} stage.classList.add('dragging'); }
    if(drag.moved){ e.preventDefault(); layout(dx); } });
  const end=e=>{ if(!drag) return; const dx=e.clientX-drag.x; const moved=drag.moved; drag=null; stage.classList.remove('dragging');
    if(moved && Math.abs(dx)>stage.clientWidth*0.12) go(cur+(dx<0?1:-1)); else layout(0); setTimeout(()=>{dragMoved=false;},50); };
  stage.addEventListener('pointerup',end); stage.addEventListener('pointercancel',end);
  document.addEventListener('keydown',e=>{ if(e.key==='ArrowLeft') go(cur-1); else if(e.key==='ArrowRight') go(cur+1); });
  window.addEventListener('resize',()=>{ layout(0); slides.forEach(sl=>sl.setP(sl.p)); });
  go(0); slides.forEach(sl=>sl.setP(sl.p));
}
(function(){ const G=(window.CASES||{}).galleries||{}; mountGallery('gallery-gen',G.generation); mountGallery('gallery-rev',G.revision); mountGallery('gallery-edit',G.editing); })();
