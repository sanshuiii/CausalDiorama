/* Scroll-driven stage curtain: closed at the top of the page, parting as the reader scrolls down. */
(function(){
  const root=document.documentElement, stage=document.getElementById('stage'), pin=document.getElementById('stage-pin');
  const curtain=document.getElementById('curtain'), marquee=document.getElementById('marquee');
  if(!stage||!pin||!curtain) return;
  const reduce=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if(reduce){ root.classList.add('no-curtain'); return; }
  const video=pin.querySelector('.video-hero'), hero=pin.querySelector('header.hero');
  let travel=0;
  function fit(){
    const vh=window.innerHeight;
    /* size the PV so title, buttons and video share the first screen where possible */
    if(video){ const room=vh-hero.offsetHeight-48; const w=Math.max(300,Math.min(960,room*16/9)); video.style.maxWidth=w+'px'; }
    travel=Math.round(vh*0.9);
    stage.style.height=(pin.offsetHeight+travel)+'px';
    curtain.style.height=vh+'px';
    /* how much of each drape stays gathered at the side once open: clear of the content */
    const vw=window.innerWidth, content=Math.max(video?video.getBoundingClientRect().width:0, Math.min(vw,860));
    const keepPx=Math.max(14,Math.min(vw*.12,(vw-content)/2-14));
    curtain.style.setProperty('--keep',(keepPx/(vw*.506)).toFixed(4));
    update();
  }
  const ease=t=>t<.5?2*t*t:1-Math.pow(-2*t+2,2)/2;
  function update(){
    const y=-stage.getBoundingClientRect().top;
    const p=Math.min(1,Math.max(0,y/travel)), e=ease(p);
    curtain.style.setProperty('--e',e.toFixed(4));
    marquee.style.opacity=Math.max(0,1-p*3.2).toFixed(3);
    marquee.style.pointerEvents=p<.25?'auto':'none';
    root.classList.toggle('curtain-open',p>=1);
  }
  function openCurtain(){ const top=stage.getBoundingClientRect().top+window.scrollY; window.scrollTo({top:top+travel,behavior:'smooth'}); }
  marquee.addEventListener('click',openCurtain);
  document.querySelectorAll('a[href="#video"]').forEach(a=>a.addEventListener('click',ev=>{ev.preventDefault(); openCurtain();}));
  let raf=0; window.addEventListener('scroll',()=>{ if(!raf) raf=requestAnimationFrame(()=>{raf=0; update();}); },{passive:true});
  window.addEventListener('resize',fit);
  if(document.fonts&&document.fonts.ready) document.fonts.ready.then(fit);
  window.addEventListener('load',fit);
  fit();
})();

/* Gentle reveal of each section as it enters the view. */
(function(){
  if(window.matchMedia('(prefers-reduced-motion: reduce)').matches||!('IntersectionObserver' in window)) return;
  const secs=[...document.querySelectorAll('body > section')];
  const io=new IntersectionObserver(es=>es.forEach(en=>{ if(en.isIntersecting){ en.target.classList.add('shown'); io.unobserve(en.target); } }),{rootMargin:'0px 0px -8% 0px'});
  secs.forEach(s=>{ s.classList.add('reveal'); io.observe(s); });
})();
