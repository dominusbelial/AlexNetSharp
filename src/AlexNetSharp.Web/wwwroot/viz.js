/* Input/animation glue only. Geometry, hit testing and pixel rendering are C#. */
window.viz = {
  reference: null, frames: 0, firstFrameMs: null,
  start(reference) {
    this.reference = reference;
    const canvas = document.getElementById('theCanvas');
    let down = false, moved = false, px = 0, py = 0, pan = false;
    const point = e => { const r = canvas.getBoundingClientRect(); return [(e.clientX-r.left)*canvas.width/r.width,(e.clientY-r.top)*canvas.height/r.height]; };
    canvas.addEventListener('pointerdown', e => { down=true;moved=false;px=e.clientX;py=e.clientY;pan=e.shiftKey||e.button===2;canvas.focus({preventScroll:true});canvas.setPointerCapture(e.pointerId); });
    canvas.addEventListener('pointermove', e => {
      if(down) { const dx=e.clientX-px,dy=e.clientY-py;if(Math.abs(dx)+Math.abs(dy)>1)moved=true;reference.invokeMethod('MoveCamera',dx,dy,pan);px=e.clientX;py=e.clientY; }
      else reference.invokeMethod('Pointer',...point(e),false);
    });
    canvas.addEventListener('pointerup', e => { if(!moved)reference.invokeMethod('Pointer',...point(e),true);down=false; });
    canvas.addEventListener('pointerleave', () => { if(!down)reference.invokeMethod('Pointer',-1,-1,false); });
    canvas.addEventListener('wheel', e => {e.preventDefault();reference.invokeMethod('Wheel',e.deltaY);},{passive:false});
    canvas.addEventListener('contextmenu', e => e.preventDefault());
    canvas.addEventListener('keydown',e=>{if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','w','a','s','d','r','+','=','-'].includes(e.key)){e.preventDefault();reference.invokeMethod('Key',e.key);}});
    const frame = () => {
      const r=canvas.getBoundingClientRect(),w=Math.max(1,Math.round(r.width)),h=Math.max(1,Math.round(r.height));
      const resized=canvas.width!==w||canvas.height!==h;
      if(resized){canvas.width=w;canvas.height=h;}
      if(!reference.invokeMethod('Frame',w,h)) return;
      this.frames++;
      if(this.firstFrameMs===null){this.firstFrameMs=performance.now();document.body.dataset.ready='true';}
      if(resized)reference.invokeMethod('Resize');
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  },
  layout(){return this.reference.invokeMethod('LayoutPoints');},
  view(){return this.reference.invokeMethod('ViewState');}
};
(async()=>{
  const paths=['JSInterop/js/JSObject','Dom/js/Window','Dom/js/Document','Dom/js/Navigator','Dom/js/Gamepad','Dom/js/Media','XHR/js/XHR','Canvas/js/Canvas','Canvas/js/CanvasGLContext','Audio/js/Audio','XR/js/XR'];
  for(const path of paths){const [pkg,...rest]=path.split('/');await new Promise((resolve,reject)=>{const s=document.createElement('script');s.src=`_content/nkast.Wasm.${pkg}/${rest.join('/')}.10.0.3.js`;s.onload=resolve;s.onerror=reject;document.body.append(s);});}
  await new Promise((resolve,reject)=>{const s=document.createElement('script');s.src='_framework/blazor.webassembly.js';s.onload=resolve;s.onerror=reject;document.body.append(s);});
})().catch(error=>{document.getElementById('app').textContent='Runtime download failed. Reload to retry.';console.error(error);});
