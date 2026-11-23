/* Tab switching (kept for any tabbed section). */
(function(){
  const IMG='static/images/', VID='static/videos/';
  const touch = matchMedia('(hover: none)').matches;
  const C=window.CASES||{};
  /* pause videos in a panel that gets hidden */
  document.querySelectorAll('.tabs').forEach(tabs=>{
    const group=tabs.dataset.group;
    tabs.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>{
      tabs.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x===b));
      document.querySelectorAll('.panel[data-group="'+group+'"]').forEach(p=>{
        const on=p.dataset.panel===b.dataset.tab; p.classList.toggle('active',on);
        if(!on) p.querySelectorAll('video').forEach(v=>v.pause()); else p.querySelectorAll('video[autoplay]').forEach(v=>v.play().catch(()=>{}));
      });
    }));
  });
})();
