/* PaddleOCR English PP-OCRv3, Apache-2.0. Rank OCR only, not a 52-card
   classifier. SWHL/RapidOCR revision 4e46440. Runs locally through ORT WASM. */
(() => {
  'use strict';
  let modelPromise;
  async function ready(){
    if(!modelPromise)modelPromise=(async()=>{
      if(!window.ort)await new Promise((resolve,reject)=>{
        const script=document.createElement('script');script.src=new URL('vendor/ort.wasm.min.js',document.baseURI).href;
        script.onload=resolve;script.onerror=()=>reject(new Error('ONNX Runtime yüklenemedi.'));document.head.appendChild(script);
      });
      ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;
      ort.env.wasm.wasmPaths=new URL('vendor/',document.baseURI).href;
      const [session,dictionary]=await Promise.all([
        ort.InferenceSession.create(new URL('models/en_PP-OCRv3_rec_infer.onnx',document.baseURI).href,{executionProviders:['wasm'],graphOptimizationLevel:'all'}),
        fetch(new URL('models/en_dict.txt',document.baseURI)).then(r=>{if(!r.ok)throw new Error('OCR sözlüğü bulunamadı.');return r.text();})
      ]);
      const chars=['',...dictionary.replace(/\r/g,'').replace(/\n$/,'').split('\n'),' '];
      return {session,chars};
    })().catch(error=>{modelPromise=null;throw error;});
    return modelPromise;
  }
  async function read(image){
    const {session,chars}=await ready();
    const height=48,width=320,actualWidth=Math.min(width,Math.ceil(height*image.width/image.height));
    const c=Object.assign(document.createElement('canvas'),{width:actualWidth,height}),ctx=c.getContext('2d');
    ctx.fillStyle='white';ctx.fillRect(0,0,c.width,c.height);ctx.drawImage(image,0,0,c.width,c.height);
    const pixels=ctx.getImageData(0,0,c.width,c.height).data,data=new Float32Array(3*height*width);
    for(let y=0;y<height;y++)for(let x=0;x<actualWidth;x++)for(let ch=0;ch<3;ch++)data[ch*height*width+y*width+x]=pixels[(y*actualWidth+x)*4+(2-ch)]/127.5-1;
    const outputs=await session.run({[session.inputNames[0]]:new ort.Tensor('float32',data,[1,3,height,width])});
    const output=outputs[session.outputNames[0]],classes=output.dims[2],steps=output.dims[1];
    if(classes!==chars.length)throw new Error(`OCR sözlük/model uyumsuzluğu: ${classes}/${chars.length}`);
    let text='',last=0,scores=[];
    for(let t=0;t<steps;t++){
      let index=0,value=-Infinity;
      for(let k=0;k<classes;k++){const v=output.data[t*classes+k];if(v>value){index=k;value=v;}}
      if(index&&index!==last){text+=chars[index];scores.push(value);}last=index;
    }
    return {text:text.trim().toUpperCase(),score:scores.length?scores.reduce((a,b)=>a+b,0)/scores.length:0};
  }
  window.CardOCR={ready,read};
})();
