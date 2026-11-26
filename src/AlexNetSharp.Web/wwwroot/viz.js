/* Input/animation glue only. Geometry, hit testing and pixel rendering are C#. */
window.viz = {
  reference: null, frames: 0, firstFrameMs: null,
  start(reference) {
    this.reference = reference;
    const canvas = document.getElementById('theCanvas');
    // Our pointer events own gestures. KNI 10.0.3's legacy window touch listeners
    // unconditionally preventDefault, even outside the canvas and on noncancelable
    // touchend events. Stop their bubble path without cancelling browser defaults.
    for(const type of ['touchstart','touchmove','touchend','touchcancel'])
      document.addEventListener(type,e=>e.stopPropagation(),{passive:true});
    let moved = false, pan = false, pinch = 0;
    const pointers = new Map();
    const point = e => { const r = canvas.getBoundingClientRect(); return [(e.clientX-r.left)*canvas.width/r.width,(e.clientY-r.top)*canvas.height/r.height]; };
    const distance = () => {const [a,b]=[...pointers.values()];return Math.hypot(a.x-b.x,a.y-b.y);};
    canvas.addEventListener('pointerdown', e => {
      pointers.set(e.pointerId,{x:e.clientX,y:e.clientY});moved=false;pan=e.shiftKey||e.button===2;
      if(pointers.size===2){pinch=distance();moved=true;}
      canvas.focus({preventScroll:true});canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', e => {
      const previous=pointers.get(e.pointerId);
      if(previous){
        pointers.set(e.pointerId,{x:e.clientX,y:e.clientY});
        if(pointers.size===2){const next=distance();if(pinch>0&&next>0)reference.invokeMethod('Wheel',-1000*Math.log(next/pinch));pinch=next;moved=true;}
        else {const dx=e.clientX-previous.x,dy=e.clientY-previous.y;if(Math.abs(dx)+Math.abs(dy)>1)moved=true;reference.invokeMethod('MoveCamera',dx,dy,pan);}
      }else reference.invokeMethod('Pointer',...point(e),false);
    });
    const release = e => {if(!moved&&pointers.size===1)reference.invokeMethod('Pointer',...point(e),true);pointers.delete(e.pointerId);pinch=0;};
    canvas.addEventListener('pointerup',release);
    canvas.addEventListener('pointercancel',e=>{pointers.delete(e.pointerId);pinch=0;moved=true;});
    canvas.addEventListener('pointerleave', () => { if(!pointers.size)reference.invokeMethod('Pointer',-1,-1,false); });
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
