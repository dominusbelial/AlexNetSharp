/* Run from the repo root. Installs are local: npm install --prefix .scratch/npm playwright. */
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const os = require('node:os');
const {spawnSync} = require('node:child_process');
const assert = require('node:assert/strict');
const {chromium} = require('../.scratch/npm/node_modules/playwright');
const root = path.resolve(__dirname, '..');
const scratch = path.join(root,'.scratch/viz1');
fs.mkdirSync(scratch,{recursive:true});
const publish = path.join(scratch,'publish');
const env = {...process.env,
 DOTNET_CLI_HOME:path.join(root,'.scratch/dotnet-home'),DOTNET_CLI_TELEMETRY_OPTOUT:'1',
 DOTNET_GENERATE_ASPNET_CERTIFICATE:'false',DOTNET_CLI_WORKLOAD_UPDATE_NOTIFY_DISABLE:'true',
 NUGET_PACKAGES:path.join(root,'.scratch/nuget-packages'),NUGET_HTTP_CACHE_PATH:path.join(root,'.scratch/nuget-http-cache'),
 TMPDIR:path.join(root,'.scratch/tmp'),XDG_RUNTIME_DIR:path.join(root,'.scratch/runtime'),
 HOME:path.join(root,'.scratch/browser-home'),
 XDG_CACHE_HOME:path.join(root,'.scratch/cache'),XDG_CONFIG_HOME:path.join(root,'.scratch/config'),XDG_DATA_HOME:path.join(root,'.scratch/data')};
for(const directory of ['.scratch/tmp','.scratch/runtime','.scratch/browser-home'])fs.mkdirSync(path.join(root,directory),{recursive:true});
process.env.TMPDIR=env.TMPDIR;
if (!process.argv.includes('--skip-publish')) {
 fs.rmSync(publish,{recursive:true,force:true});
 const result=spawnSync(path.join(os.homedir(),'.dotnet/dotnet'),['publish','src/AlexNetSharp.Web','-c','Release','-o',publish,'--disable-build-servers'],{cwd:root,env,encoding:'utf8'});
 fs.writeFileSync(path.join(scratch,'browser-publish.log'),result.stdout+result.stderr);
 assert.equal(result.status,0,'trimmed Release publish');
 assert(!/\bwarning\b|\berror [A-Z]+\d+/i.test(result.stdout+result.stderr),'publish has no warnings/errors');
}
const wwwroot=path.join(publish,'wwwroot');
const mime={'.html':'text/html','.css':'text/css','.js':'text/javascript','.json':'application/json','.wasm':'application/wasm','.bin':'application/octet-stream','.md':'text/plain'};
const server=http.createServer((req,res)=>{
 let pathname=decodeURIComponent(req.url.split('?')[0]);
 if(pathname==='/')pathname='/index.html';
 const file=path.resolve(wwwroot,'.'+pathname);
 if(!file.startsWith(wwwroot+path.sep)||!fs.existsSync(file)||!fs.statSync(file).isFile()){res.writeHead(404);res.end();return;}
 res.setHeader('Content-Type',mime[path.extname(file)]||'application/octet-stream');
 fs.createReadStream(file).pipe(res);
});
function sizeTree(directory){return fs.readdirSync(directory,{withFileTypes:true}).reduce((sum,e)=>sum+(e.isDirectory()?sizeTree(path.join(directory,e.name)):fs.statSync(path.join(directory,e.name)).size),0);}
(async()=>{
 await new Promise(resolve=>server.listen(0,Object.values(os.networkInterfaces()).flat().find(i=>i.internal&&i.family==='IPv4').address,resolve));
 const address=server.address();const origin=`http://${address.address}:${address.port}`;
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH||'/usr/bin/google-chrome',args:['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'],env});
 const results=[];
 try {
  for(const viewport of [{width:1280,height:800},{width:390,height:844}]){
   const context=await browser.newContext({viewport,deviceScaleFactor:1});const page=await context.newPage();
   const errors=[];
   page.on('pageerror',error=>{errors.push(error.message);console.log('PAGE ERROR',error.message);});
   page.on('console',message=>{if(message.type()==='error'){errors.push(message.text());console.log('CONSOLE ERROR',message.text());}});
   page.on('response',response=>{if(response.status()>=400)errors.push('HTTP '+response.status()+' '+new URL(response.url()).pathname);});
   await page.goto(origin,{waitUntil:'domcontentloaded'});
   await page.waitForSelector('body[data-ready="true"]',{timeout:120000});
   await page.waitForFunction(()=>window.viz.frames>5);
   assert.match(await page.locator('#untrained-banner').innerText(),/untrained placeholder weights/i);
   assert.match(await page.locator('#untrained-banner').innerText(),/not bitwise historical parity/i);
   const points=await page.evaluate(()=>window.viz.layout());assert.equal(points.length,49);
   const canvas=page.locator('#theCanvas');let bounds=await canvas.boundingBox();
   const point=name=>points.find(p=>p.name===name);
   const conv1=point('conv1a');
   await page.mouse.move(bounds.x+conv1.x,bounds.y+conv1.y);
   await page.waitForSelector('#tooltip');assert.match(await page.locator('#tooltip').innerText(),/conv1a/);
   await page.mouse.click(bounds.x+conv1.x,bounds.y+conv1.y);
   await page.waitForSelector('[data-card="conv1a"]');
   const conv2=point('conv2a');
   await page.mouse.click(bounds.x+conv2.x,bounds.y+conv2.y);
   await page.waitForSelector('[data-card="conv2a"]');assert.equal(await page.locator('.explain-card').count(),2);
   await page.mouse.click(bounds.x+conv1.x,bounds.y+conv1.y);
   await page.mouse.move(bounds.x+5,bounds.y+5);
   // Fit the entire canvas before selecting a real atlas pixel (channel 0, x=3, y=4).
   await canvas.scrollIntoViewIfNeeded();bounds=await canvas.boundingBox();
   let state=await page.evaluate(()=>window.viz.view());let r=state.grid;
   const cols=Math.ceil(Math.sqrt(48*1.6)),rows=Math.ceil(48/cols),tile=57;
   await page.mouse.click(bounds.x+r.x+r.width*(4.5/(cols*tile)),bounds.y+r.y+r.height*(5.5/(rows*tile)));
   await page.waitForFunction(()=>window.viz.view().field.contributing.x0===12&&window.viz.view().field.contributing.y0===16);
   assert.deepEqual(await page.evaluate(()=>window.viz.view().pixel),{channel:0,x:3,y:4});
   await page.locator('#pixel-x').fill('54');await page.locator('#pixel-x').press('Tab');
   await page.waitForFunction(()=>window.viz.view().field.contributing.x0===216);
   assert.match(await page.locator('#receptive-field').innerText(),/216\.\.223/);
   const pixels=await page.evaluate(()=>{
    const canvas=document.getElementById('theCanvas');window.viz.reference.invokeMethod('Frame',canvas.width,canvas.height);
    const gl=canvas.getContext('webgl2')||canvas.getContext('webgl');const data=new Uint8Array(canvas.width*canvas.height*4);
    gl.readPixels(0,0,canvas.width,canvas.height,gl.RGBA,gl.UNSIGNED_BYTE,data);
    let nonBackground=0,mint=0,yellow=0;
    for(let i=0;i<data.length;i+=4){if(data[i]!==11||data[i+1]!==20||data[i+2]!==33)nonBackground++;if(data[i]>20&&data[i]<85&&data[i+1]>80&&data[i+1]>data[i]*1.6&&data[i+2]>data[i]*1.4&&data[i+3]>0)mint++;if(data[i]>245&&data[i+1]>205&&data[i+1]<230&&data[i+2]>100&&data[i+2]<135)yellow++;}
    return {nonBackground,mint,yellow};
   });
   assert(pixels.nonBackground>500,'canvas draws');assert(pixels.mint>50,'activation textures draw');assert(pixels.yellow>4,'receptive-field yellow outline draws');
   await page.locator('[data-card="conv1a"] > details > summary').click();
   assert.match(await page.locator('[data-card="conv1a"]').innerText(),/Snapshot output\[0\]/);
   // Camera keyboard, wheel and pointer motion all change the pure projected coordinates.
   await canvas.scrollIntoViewIfNeeded();bounds=await canvas.boundingBox();
   await canvas.focus();await page.keyboard.press('ArrowRight');
   const moved=await page.evaluate(()=>window.viz.layout());assert.notEqual(moved.find(p=>p.name==='conv1a').x,conv1.x);
   await page.keyboard.press('r');
   const stackPixel=await page.evaluate(p=>{
    const c=document.getElementById('theCanvas');window.viz.reference.invokeMethod('Frame',c.width,c.height);
    const gl=c.getContext('webgl2')||c.getContext('webgl'),v=new Uint8Array(4);gl.readPixels(Math.round(p.x),Math.round(c.height-p.y),1,1,gl.RGBA,gl.UNSIGNED_BYTE,v);return [...v];
   },conv1);
   assert(stackPixel[1]>100&&stackPixel[1]>stackPixel[0]*1.3,'3D selected box stays aligned after camera reset');
   const overflow=await page.evaluate(()=>{
    const vw=document.documentElement.clientWidth;
    const clipped=[...document.querySelectorAll('.explain-card,.aside-heading')].filter(e=>{const r=e.getBoundingClientRect();return r.left<-1||r.right>vw+1;}).length;
    return {page:document.documentElement.scrollWidth-vw,clipped};
   });
   assert.equal(overflow.page<=1&&overflow.clipped===0,true,`no horizontal overflow at ${viewport.width}px: ${JSON.stringify(overflow)}`);
   const screenshot=path.join(scratch,`browser-${viewport.width}.png`);await page.screenshot({path:screenshot,fullPage:true});
   assert(fs.statSync(screenshot).size<400_000,'report screenshot under 400 KB');
   await page.locator('#filter-tab').click();
   assert.equal(await page.evaluate(()=>window.viz.view().selected),'conv1a');
   assert.match(await page.locator('.view-heading').innerText(),/96 RGB FILTERS/);
   const filterColors=await page.evaluate(()=>{
    const c=document.getElementById('theCanvas');window.viz.reference.invokeMethod('Frame',c.width,c.height);
    const gl=c.getContext('webgl2')||c.getContext('webgl'),r=window.viz.view().grid;
    const data=new Uint8Array(r.width*r.height*4);gl.readPixels(r.x,c.height-r.y-r.height,r.width,r.height,gl.RGBA,gl.UNSIGNED_BYTE,data);
    const colors=new Set();for(let i=0;i<data.length;i+=4)colors.add(`${data[i]},${data[i+1]},${data[i+2]}`);return colors.size;
   });
   assert(filterColors>32,'RGB filter textures draw');
   assert.deepEqual(errors,[],'no browser console, runtime or HTTP errors');
   results.push({viewport,firstFrameMs:await page.evaluate(()=>window.viz.firstFrameMs),frames:await page.evaluate(()=>window.viz.frames),pixels,stackPixel,filterColors,screenshotBytes:fs.statSync(screenshot).size,consoleErrors:errors.length});
   await context.close();
  }
 } finally {await browser.close();await new Promise(resolve=>server.close(resolve));}
 const result={passed:true,publishBytes:sizeTree(publish),webRootBytes:sizeTree(wwwroot),snapshotBytes:sizeTree(path.join(wwwroot,'replay')),results};
 fs.writeFileSync(path.join(scratch,'browser-proof.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
})().catch(error=>{console.error(error);server.close();process.exitCode=1;});
