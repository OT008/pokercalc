/* All processing stays on the device. No image upload, TextDetector, or model API. */
(() => {
  'use strict';
  let runtimePromise, templatesPromise;
  const median = values => { const a=[...values].sort((a,b)=>a-b);return a.length?a[Math.floor(a.length/2)]:0; };
  const distance = (a,b) => Math.hypot(a.x-b.x,a.y-b.y);
  const mean = a => a.reduce((s,v)=>s+v,0)/Math.max(1,a.length);
  const canvas = (w,h) => Object.assign(document.createElement('canvas'),{width:w,height:h});
  const release = (...items) => items.forEach(item=>item&&item.delete());

  async function ready(){
    if(!runtimePromise)runtimePromise=new Promise((resolve,reject)=>{
      const script=document.createElement('script');
      const timer=setTimeout(()=>{runtimePromise=null;reject(new Error('OpenCV yüklenemedi. Bağlantıyı kontrol edip tekrar deneyin.'));},45000);
      script.src=new URL('vendor/opencv.js',document.baseURI).href;
      script.onerror=()=>{clearTimeout(timer);runtimePromise=null;reject(new Error('OpenCV dosyası yüklenemedi.'));};
      script.onload=()=>{
        try{
          let lib=window.cv;
          const finish=loaded=>{clearTimeout(timer);window.cv=loaded;resolve();};
          // OpenCV 4.10 exposes a legacy self-returning thenable. Awaiting that
          // object would loop forever; resolve our promise without its value.
          if(lib.Mat)finish(lib);
          else if(typeof lib.then==='function')lib.then(finish);
          else lib.onRuntimeInitialized=()=>finish(lib);
        }catch(error){clearTimeout(timer);runtimePromise=null;reject(error);}
      };
      document.head.appendChild(script);
    });
    return runtimePromise;
  }

  async function templates(){
    if(!templatesPromise)templatesPromise=fetch(new URL('rank-templates.json',document.baseURI)).then(r=>{
      if(!r.ok)throw new Error('Kart şablonları yüklenemedi.');return r.json();
    }).then(rows=>rows.map(t=>{const mask=Uint8Array.from(t.bits,ch=>Number(ch));return {...t,mask,dist:distanceMap(mask)};})).catch(error=>{templatesPromise=null;throw error;});
    return templatesPromise;
  }

  // Return EXIF for diagnostics. createImageBitmap applies all 8 EXIF transforms,
  // including mirrored orientations; never rotate it a second time from EXIF.
  async function exifOrientation(file){
    const view=new DataView(await file.slice(0,262144).arrayBuffer());
    if(view.byteLength<4||view.getUint16(0)!==0xffd8)return 1;
    for(let p=2;p+4<view.byteLength;){
      const marker=view.getUint16(p);const length=view.getUint16(p+2);
      if(marker===0xffe1&&p+10<view.byteLength&&view.getUint32(p+4)===0x45786966){
        const start=p+10,little=view.getUint16(start)===0x4949;
        const ifd=start+view.getUint32(start+4,little);
        if(ifd+2>=view.byteLength)return 1;
        const count=view.getUint16(ifd,little);
        for(let i=0;i<count;i++){
          const entry=ifd+2+i*12;if(entry+12>view.byteLength)break;
          if(view.getUint16(entry,little)===0x0112)return view.getUint16(entry+8,little);
        }
      }
      if(length<2)break;p+=length+2;
    }
    return 1;
  }

  function orderCorners(points){
    const center={x:mean(points.map(p=>p.x)),y:mean(points.map(p=>p.y))};
    const p=[...points].sort((a,b)=>Math.atan2(a.y-center.y,a.x-center.x)-Math.atan2(b.y-center.y,b.x-center.x));
    let first=p.reduce((best,v,i)=>v.x+v.y<p[best].x+p[best].y?i:best,0);
    let ordered=Array.from({length:4},(_,i)=>p[(first+i)%4]);
    if(distance(ordered[0],ordered[1])>distance(ordered[1],ordered[2]))ordered=[ordered[1],ordered[2],ordered[3],ordered[0]];
    return ordered;
  }

  function detect(source,diagnostics){
    const cv=window.cv,src=cv.imread(source),gray=new cv.Mat(),blur=new cv.Mat();
    cv.cvtColor(src,gray,cv.COLOR_RGBA2GRAY);cv.GaussianBlur(gray,blur,new cv.Size(5,5),0);
    const area=source.width*source.height,candidates=[],reasons={};
    const reject=reason=>{reasons[reason]=(reasons[reason]||0)+1;};
    function contours(mask,method){
      const list=new cv.MatVector(),hierarchy=new cv.Mat();
      cv.findContours(mask,list,hierarchy,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE);
      for(let i=0;i<list.size();i++){
        const c=list.get(i);const a=cv.contourArea(c);
        if(a<area*.009||a>area*.20){reject('alan');c.delete();continue;}
        const rect=cv.minAreaRect(c),w=rect.size.width,h=rect.size.height;
        const long=Math.max(w,h),short=Math.min(w,h),ratio=long/short,fill=a/(w*h);
        if(ratio<1.2||ratio>1.9||fill<.68){reject('oran/dörtgen doluluk');c.delete();continue;}
        const approx=new cv.Mat();cv.approxPolyDP(c,approx,.025*cv.arcLength(c,true),true);
        let points;
        if(approx.rows===4&&cv.isContourConvex(approx)){
          points=Array.from({length:4},(_,j)=>({x:approx.data32S[j*2],y:approx.data32S[j*2+1]}));
        }else points=cv.RotatedRect.points(rect);
        const center={x:mean(points.map(p=>p.x)),y:mean(points.map(p=>p.y))};
        candidates.push({points,center,long,short,fill,area:a,method});
        release(c,approx);
      }
      release(list,hierarchy);
    }
    const edge=new cv.Mat(),mask=new cv.Mat(),work=new cv.Mat(),channel=new cv.Mat(),channels=new cv.MatVector();
    cv.split(src,channels);
    // Multichannel edges: the edge can disappear in gray while remaining visible
    // in a color channel on a patterned background.
    for(let ch=-1;ch<3;ch++){
      if(ch<0)blur.copyTo(channel);else {const part=channels.get(ch);cv.GaussianBlur(part,channel,new cv.Size(9,9),0);part.delete();}
      for(const [lo,hi] of [[12,35],[25,75],[50,140]]){
        cv.Canny(channel,edge,lo,hi);
        for(const size of [3,7,11]){
          const kernel=cv.Mat.ones(size,size,cv.CV_8U);
          cv.morphologyEx(edge,mask,cv.MORPH_CLOSE,kernel);contours(mask,`kenar:${ch}/${lo}/${size}`);kernel.delete();
        }
      }
      // These masks add proposals only. Acceptance still requires card geometry,
      // common dimensions, and corner/edge evidence; either polarity is accepted.
      for(const threshold of [45,55,65,75,85,95,105,115,130,145,160]){
        cv.threshold(channel,edge,threshold,255,cv.THRESH_BINARY_INV);
        for(const size of [5,11,21]){
          const kernel=cv.Mat.ones(size,size,cv.CV_8U);
          cv.morphologyEx(edge,work,cv.MORPH_CLOSE,kernel);cv.morphologyEx(work,mask,cv.MORPH_OPEN,kernel);
          contours(mask,`kontrast:${ch}/${threshold}/${size}`);kernel.delete();
        }
      }
    }
    const rgb=new cv.Mat(),hsv=new cv.Mat();cv.cvtColor(src,rgb,cv.COLOR_RGBA2RGB);cv.cvtColor(rgb,hsv,cv.COLOR_RGB2HSV);
    for(const sat of [60,80,100,120,140])for(const val of [90,120,150,180]){
      const low=new cv.Mat(hsv.rows,hsv.cols,hsv.type(),new cv.Scalar(0,0,0));
      const high=new cv.Mat(hsv.rows,hsv.cols,hsv.type(),new cv.Scalar(180,sat,val));cv.inRange(hsv,low,high,edge);release(low,high);
      for(const size of [7,15,25]){
        const kernel=cv.Mat.ones(size,size,cv.CV_8U);cv.morphologyEx(edge,work,cv.MORPH_CLOSE,kernel);cv.morphologyEx(work,mask,cv.MORPH_OPEN,kernel);
        contours(mask,'renk+geometri');kernel.delete();
      }
    }
    release(src,gray,blur,edge,mask,work,channel,channels,rgb,hsv);
    diagnostics.rejected=reasons;diagnostics.rawCandidates=candidates.length;
    if(!candidates.length)return [];
    const bins=new Map();
    candidates.forEach(c=>{const key=`${Math.round(Math.log(c.long)/.15)},${Math.round(Math.log(c.short)/.15)}`;if(!bins.has(key))bins.set(key,[]);bins.get(key).push(c);});
    const mode=[...bins.values()].sort((a,b)=>b.length-a.length)[0];
    const typical={long:median(mode.map(c=>c.long)),short:median(mode.map(c=>c.short))};
    diagnostics.typicalSize=typical;
    const quality=c=>Math.abs(Math.log(c.long/typical.long))+Math.abs(Math.log(c.short/typical.short))+.1*(1-c.fill);
    const selected=[];
    for(const c of candidates.sort((a,b)=>quality(a)-quality(b))){
      if(Math.abs(Math.log(c.long/typical.long))>.25||Math.abs(Math.log(c.short/typical.short))>.25){reject('ortak kart boyutu');continue;}
      if(selected.some(s=>distance(s.center,c.center)<typical.short*.45)){reject('aynı kartın iç/çift konturu');continue;}
      selected.push(c);
    }
    supplementFromCorners(source,selected,typical,diagnostics);
    return selected.map((c,i)=>{
      const ordered=orderCorners(c.points);
      const measuredLong=Math.max((distance(ordered[0],ordered[1])+distance(ordered[2],ordered[3]))/2,(distance(ordered[1],ordered[2])+distance(ordered[3],ordered[0]))/2);
      const factor=Math.max(1,Math.min(1.22,typical.long/measuredLong));
      const points=orderCorners(c.points.map(p=>({x:c.center.x+(p.x-c.center.x)*factor,y:c.center.y+(p.y-c.center.y)*factor})));
      const x=Math.min(...points.map(p=>p.x)),y=Math.min(...points.map(p=>p.y));
      return {id:`vision_${i}`,index:i+1,points,x,y,w:Math.max(...points.map(p=>p.x))-x,h:Math.max(...points.map(p=>p.y))-y,center:c.center,detected:true,rank:'',suit:'',method:c.method,confidence:{rank:0,suit:0},size:typical};
    });
  }

  function supplementFromCorners(source,selected,size,diagnostics){
    const cv=window.cv,data=source.getContext('2d').getImageData(0,0,source.width,source.height),mask=cv.Mat.zeros(source.height,source.width,cv.CV_8UC1);
    for(let p=0;p<mask.data.length;p++){
      const r=data.data[p*4],g=data.data[p*4+1],b=data.data[p*4+2];
      mask.data[p]=((r>100&&r>g*1.35&&r>b*1.05)||(r>160&&g>160&&b>150&&Math.max(r,g,b)-Math.min(r,g,b)<65))?255:0;
    }
    const kernel=cv.Mat.ones(2,2,cv.CV_8U);cv.morphologyEx(mask,mask,cv.MORPH_OPEN,kernel);
    const list=new cv.MatVector(),hierarchy=new cv.Mat();cv.findContours(mask,list,hierarchy,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE);
    const glyphs=[];
    for(let i=0;i<list.size();i++){
      const c=list.get(i),box=cv.boundingRect(c);
      if(box.width>size.short*.032&&box.height>size.short*.032&&box.width<size.short*.17&&box.height<size.short*.17&&cv.contourArea(c)>size.short*size.short*.001)
        glyphs.push({x:box.x+box.width/2,y:box.y+box.height/2});
      c.delete();
    }
    release(mask,kernel,list,hierarchy);
    // The local direction comes from card edges, never absolute photo axes.
    const angles=selected.map(c=>{const p=orderCorners(c.points);const dx=p[2].x-p[1].x,dy=p[2].y-p[1].y;return Math.atan2(dy,dx);});
    const angle=median(angles);const u={x:Math.cos(angle),y:Math.sin(angle)},v={x:-u.y,y:u.x};
    const project=p=>({x:p.x*u.x+p.y*u.y,y:p.x*v.x+p.y*v.y});
    const blocks=[];
    for(let i=0;i<glyphs.length;i++)for(let j=i+1;j<glyphs.length;j++){
      const a=glyphs[i],b=glyphs[j],pa=project(a),pb=project(b),d=distance(a,b);
      if(d>size.short*.06&&d<size.short*.185&&Math.abs(pa.y-pb.y)<size.short*.08){
        const center={x:(a.x+b.x)/2,y:(a.y+b.y)/2};
        if(!blocks.some(block=>distance(block,center)<size.short*.12))blocks.push(center);
      }
    }
    diagnostics.cornerBlocks=blocks;
    for(let i=0;i<blocks.length;i++)for(let j=i+1;j<blocks.length;j++){
      const a=project(blocks[i]),b=project(blocks[j]);
      const dl=Math.abs(a.x-b.x),ds=Math.abs(a.y-b.y);
      if(dl<size.long*.57||dl>size.long*.95||ds<size.short*.60||ds>size.short*.98)continue;
      const center={x:(blocks[i].x+blocks[j].x)/2,y:(blocks[i].y+blocks[j].y)/2};
      if(selected.some(c=>distance(c.center,center)<size.short*.30))continue;
      let dark=0,count=0;
      for(let a=-.35;a<=.35;a+=.1)for(let b=-.35;b<=.35;b+=.1){
        const x=Math.round(center.x+u.x*a*size.long+v.x*b*size.short),y=Math.round(center.y+u.y*a*size.long+v.y*b*size.short);
        if(x<0||y<0||x>=source.width||y>=source.height)continue;
        const k=(y*source.width+x)*4;const brightness=(data.data[k]+data.data[k+1]+data.data[k+2])/3;dark+=brightness<145?1:0;count++;
      }
      if(dark/Math.max(1,count)<.80)continue;
      const points=[[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]].map(([a,b])=>({x:center.x+u.x*a*size.long+v.x*b*size.short,y:center.y+u.y*a*size.long+v.y*b*size.short}));
      selected.push({points,center,long:size.long,short:size.short,fill:1,method:'karşılıklı köşe + ortak kart geometrisi'});
    }
  }

  function rotatePoint(p,turn,w,h){
    return turn===0?{x:p.x,y:p.y}:turn===1?{x:h-p.y,y:p.x}:turn===2?{x:w-p.x,y:h-p.y}:{x:p.y,y:w-p.x};
  }

  function group(cards,width,height,diagnostics){
    const hypotheses=[];
    for(let turn=0;turn<4;turn++){
      const positions=cards.map(card=>({card,...rotatePoint(card.center,turn,width,height)}));
      const short=median(cards.map(c=>c.size.short));
      const rows=[];
      for(const seed of positions){
        const row=positions.filter(p=>Math.abs(p.y-seed.y)<short*.60).sort((a,b)=>a.x-b.x);
        if(row.length!==5)continue;
        if(rows.some(r=>r.map(p=>p.card.id).join()===row.map(p=>p.card.id).join()))continue;
        rows.push(row);
      }
      for(const row of rows){
        const boardY=mean(row.map(p=>p.y)),gap=row.slice(1).map((p,i)=>p.x-row[i].x);
        const spacing=mean(gap),cv=Math.sqrt(mean(gap.map(g=>(g-spacing)**2)))/spacing;
        const residual=Math.max(...row.map(p=>Math.abs(p.y-boardY)))/short;
        if(cv>.40||residual>.55||Math.min(...gap)<short*.5||Math.max(...gap)>short*2)continue;
        const rest=positions.filter(p=>!row.includes(p)).sort((a,b)=>a.x-b.x);
        if(rest.length<2||rest.some(p=>boardY-p.y<short*.80))continue;
        const groups=[];
        for(const p of rest){
          const last=groups[groups.length-1];
          if(last&&last.length===1&&distance(last[0],p)<short*1.85)last.push(p);else groups.push([p]);
        }
        const singles=groups.filter(g=>g.length===1).length;
        let separation=true;
        for(let i=1;i<groups.length;i++){
          const previous=groups[i-1],next=groups[i];
          const between=next[0].x-previous[previous.length-1].x;
          const internal=Math.max(...groups.filter(g=>g.length===2).map(g=>distance(g[0],g[1])),short*.7);
          if(between<internal*1.28)separation=false;
        }
        if(!separation||singles>1)continue;
        const score=1-cv*.5-residual*.3-singles*.15;
        hypotheses.push({turn,score,community:row.map(p=>p.card),groups:groups.map(g=>g.map(p=>p.card))});
      }
    }
    hypotheses.sort((a,b)=>b.score-a.score);
    diagnostics.orientations=hypotheses.map(h=>({degrees:h.turn*90,score:h.score,community:h.community.map(c=>c.index),groups:h.groups.map(g=>g.map(c=>c.index))}));
    const best=hypotheses[0];
    if(!best)return {reliable:false,turn:null,groups:[],community:[],unassigned:[...cards]};
    if(hypotheses[1]&&hypotheses[1].turn!==best.turn&&best.score-hypotheses[1].score<.05)return {reliable:false,turn:null,groups:[],community:[],unassigned:[...cards]};
    return {...best,reliable:true,unassigned:[]};
  }

  function warp(source,card){
    const cv=window.cv,src=cv.imread(source),output=new cv.Mat();
    const from=cv.matFromArray(4,1,cv.CV_32FC2,card.points.flatMap(p=>[p.x,p.y]));
    const to=cv.matFromArray(4,1,cv.CV_32FC2,[0,0,239,0,239,335,0,335]);
    const transform=cv.getPerspectiveTransform(from,to);
    cv.warpPerspective(src,output,transform,new cv.Size(240,336),cv.INTER_CUBIC,cv.BORDER_REPLICATE);
    const result=canvas(240,336);cv.imshow(result,output);release(src,output,from,to,transform);return result;
  }

  function rotate(source,turn){
    const result=canvas(turn%2?source.height:source.width,turn%2?source.width:source.height),ctx=result.getContext('2d');
    ctx.translate(result.width/2,result.height/2);ctx.rotate(turn*Math.PI/2);ctx.drawImage(source,-source.width/2,-source.height/2);return result;
  }

  function components(mask,w,h){
    const seen=new Uint8Array(mask.length),stack=new Int32Array(mask.length),result=[];
    for(let start=0;start<mask.length;start++){
      if(!mask[start]||seen[start])continue;
      let count=0,minX=w,maxX=0,minY=h,maxY=0,top=0;stack[top++]=start;seen[start]=1;
      while(top){
        const p=stack[--top],x=p%w,y=Math.floor(p/w);count++;minX=Math.min(minX,x);maxX=Math.max(maxX,x);minY=Math.min(minY,y);maxY=Math.max(maxY,y);
        for(const n of [p-1,p+1,p-w,p+w]){if(n<0||n>=mask.length||seen[n]||!mask[n]||Math.abs(n%w-x)>1)continue;seen[n]=1;stack[top++]=n;}
      }
      result.push({x:minX,y:minY,w:maxX-minX+1,h:maxY-minY+1,count});
    }
    return result;
  }

  function normalize(mask,w,box){
    const result=new Uint8Array(32*48);
    for(let y=0;y<48;y++)for(let x=0;x<32;x++){
      const sx=box.x+Math.min(box.w-1,Math.floor((x+.5)*box.w/32)),sy=box.y+Math.min(box.h-1,Math.floor((y+.5)*box.h/48));
      result[y*32+x]=mask[sy*w+sx];
    }
    return result;
  }

  function distanceMap(mask){
    const d=Float32Array.from(mask,v=>v?0:100);
    for(let y=0;y<48;y++)for(let x=0;x<32;x++){
      const i=y*32+x;if(x)d[i]=Math.min(d[i],d[i-1]+1);if(y)d[i]=Math.min(d[i],d[i-32]+1);if(x&&y)d[i]=Math.min(d[i],d[i-33]+1.414);if(x<31&&y)d[i]=Math.min(d[i],d[i-31]+1.414);
    }
    for(let y=47;y>=0;y--)for(let x=31;x>=0;x--){
      const i=y*32+x;if(x<31)d[i]=Math.min(d[i],d[i+1]+1);if(y<47)d[i]=Math.min(d[i],d[i+32]+1);if(x<31&&y<47)d[i]=Math.min(d[i],d[i+33]+1.414);if(x&&y<47)d[i]=Math.min(d[i],d[i+31]+1.414);
    }
    return d;
  }

  function match(mask,library,kind,aspect){
    const d=distanceMap(mask),scores=new Map();
    const topMass=mean(Array.from({length:6},(_,y)=>mean(Array.from({length:32},(_,x)=>mask[y*32+x]))));
    const holes=components(Uint8Array.from(mask,v=>1-v),32,48).filter(b=>b.x>0&&b.y>0&&b.x+b.w<32&&b.y+b.h<48&&b.count>10).length;
    for(const t of library.filter(t=>t.kind===kind)){
      let sumA=0,sumB=0,countA=0,countB=0,intersection=0,union=0;
      for(let i=0;i<mask.length;i++){
        if(mask[i]){sumA+=t.dist[i];countA++;}if(t.mask[i]){sumB+=d[i];countB++;}if(mask[i]&&t.mask[i])intersection++;if(mask[i]||t.mask[i])union++;
      }
      const chamfer=(sumA/Math.max(1,countA)+sumB/Math.max(1,countB))/2;
      let score=.65*Math.exp(-chamfer/2)+.35*intersection/Math.max(1,union);
      if(kind==='suit'&&t.label==='S')score-=Math.max(0,topMass-.18)*.8;
      if(kind==='suit'&&t.label==='C')score-=Math.max(0,.22-topMass)*.8;
      if(kind==='rank'&&t.label==='8'&&holes===1)score-=.10;
      if(kind==='rank'&&aspect&&t.aspect)score-=.28*Math.abs(Math.log(aspect/t.aspect));
      if(!scores.has(t.label)||scores.get(t.label)<score)scores.set(t.label,score);
    }
    return [...scores].map(([label,score])=>({label,score})).sort((a,b)=>b.score-a.score);
  }

  function readCorner(source,library){
    const readings=[];
    for(const w of [28,36,44,52,64]){
    const h=112,roi=canvas(w,h);roi.getContext('2d').drawImage(source,1,1,w,h,0,0,w,h);
    const pixels=roi.getContext('2d').getImageData(0,0,w,h).data;
    for(const polarity of ['red','light','dark'])for(const level of [0,1,2]){
      const mask=new Uint8Array(w*h);
      for(let p=0;p<mask.length;p++){
        const r=pixels[p*4],g=pixels[p*4+1],b=pixels[p*4+2];
        const contrast=r-Math.max(g,b),bright=Math.min(r,g,b),max=Math.max(r,g,b);
        mask[p]=polarity==='red'?(r>90&&r-g>35+level*10&&contrast>15+level*15?1:0):polarity==='light'?(bright>130+level*20&&max-bright<80?1:0):(max<135-level*20?1:0);
      }
      for(let x=0;x<w;x++){
        let run=0;
        for(let y=0;y<=h;y++){
          if(y<h&&mask[y*w+x])run++;
          else{if(run>=50)for(let yy=y-run;yy<y;yy++)mask[yy*w+x]=0;run=0;}
        }
      }
      const boxes=components(mask,w,h).filter(b=>b.w>=6&&b.h>=12&&b.h<55&&b.w<=40&&b.count>=35&&b.x+b.w<w-1).sort((a,b)=>a.y-b.y);
      for(const rankBox of boxes.filter(b=>b.y<40&&b.h>20)){
        const ranks=match(normalize(mask,w,rankBox),library,'rank',rankBox.w/rankBox.h);
        const suitMask=new Uint8Array(mask.length);
        const cx=rankBox.x+rankBox.w/2,top=rankBox.y+rankBox.h+1,bottom=Math.min(h-1,top+40);
        for(let y=top;y<bottom;y++)for(let x=Math.max(0,Math.floor(cx-20));x<Math.min(w,cx+20);x++)suitMask[y*w+x]=mask[y*w+x];
        // Card art frames sometimes touch the tiny index symbol. Remove only
        // long vertical frame runs inside this suit region, not the rank.
        for(let x=0;x<w;x++){
          let run=0;
          for(let y=top;y<=bottom;y++){
            if(y<bottom&&suitMask[y*w+x])run++;
            else{if(run>=30)for(let yy=y-run;yy<y;yy++)suitMask[yy*w+x]=0;run=0;}
          }
        }
        const suitBoxes=components(suitMask,w,h).filter(b=>b.h>=10&&b.h<=38&&b.w>=6&&b.count>=25&&Math.abs(b.x+b.w/2-cx)<12);
        if(!suitBoxes.length)readings.push({ranks,suits:[{label:'',score:0},{label:'',score:0}],score:ranks[0].score*.5,polarity,rankBox,corner:roi.toDataURL('image/png')});
        for(const suitBox of suitBoxes){
          const suitLibrary=library.filter(t=>t.kind!=='suit'||(polarity==='red'?['H','D']:['S','C']).includes(t.label));
          const suits=match(normalize(suitMask,w,suitBox),suitLibrary,'suit');
          const score=(ranks[0].score+suits[0].score)/2;
          readings.push({ranks,suits,score,polarity,rankBox,suitBox,corner:roi.toDataURL('image/png')});
        }
      }
    }
    }
    return readings.sort((a,b)=>b.score-a.score);
  }

  async function recognize(source,cards,diagnostics,onProgress){
    const library=await templates();
    let ocrAvailable=true;
    try{await CardOCR.ready();diagnostics.ocrModel='PP-OCRv3 English ONNX, loaded';}
    catch(error){ocrAvailable=false;diagnostics.ocrModelError=error.message;}
    const readOCR=async image=>{
      try{return await CardOCR.read(image);}
      catch(error){diagnostics.ocrInferenceError=error.message;return {text:'',score:0};}
    };
    for(let i=0;i<cards.length;i++){
      const card=cards[i],flat=warp(source,card),views=[];
      for(let turn=0;turn<4;turn++){
        const oriented=rotate(flat,turn);
        for(const r of readCorner(oriented,library))views.push({...r,turn});
      }
      views.sort((a,b)=>b.score-a.score);
      const best=views[0];
      card.rectified=flat.toDataURL('image/jpeg',.85);
      card.readings=views.slice(0,8).map(r=>({turn:r.turn,score:r.score,rank:r.ranks.slice(0,3),suit:r.suits.slice(0,2),corner:r.corner,polarity:r.polarity}));
      if(best){
        const rankView=[...views].sort((a,b)=>b.ranks[0].score-a.ranks[0].score)[0];
        const suitView=[...views].filter(v=>v.ranks[0].score>.73).sort((a,b)=>b.suits[0].score-a.suits[0].score)[0]||best;
        const rank=rankView.ranks[0],suit=suitView.suits[0];
        card.confidence={rank:rank.score,suit:suit.score,margin:rank.score-rankView.ranks[1].score};
        if(rank.score>=.78&&rank.score-rankView.ranks[1].score>=.025)card.rank=rank.label;
        if(suit.score>=.79&&suit.score-suitView.suits[1].score>=.04)card.suit=suit.label;
      }
      if(ocrAvailable){
        const ocrViews=[];
        for(let turn=0;turn<4;turn++){
          const view=views.filter(v=>v.turn===turn).sort((a,b)=>b.ranks[0].score-a.ranks[0].score)[0];
          if(!view||view.ranks[0].score<.68)continue;
          const oriented=rotate(flat,turn),box=view.rankBox;
          const crop=canvas(box.w+6,box.h+6),ctx=crop.getContext('2d');
          ctx.fillStyle='white';ctx.fillRect(0,0,crop.width,crop.height);
          ctx.drawImage(oriented,box.x+1,box.y+1,box.w,box.h,3,3,box.w,box.h);
          const input=ctx.getImageData(0,0,crop.width,crop.height);
          const softCrop=canvas(crop.width,crop.height),softCtx=softCrop.getContext('2d'),soft=softCtx.createImageData(crop.width,crop.height);
          for(let p=0;p<input.data.length;p+=4){
            const r=input.data[p],g=input.data[p+1],b=input.data[p+2];
            // The selected mask already identified the foreground polarity.
            const ink=view.polarity==='red'?(r>90&&r-g>35&&r-Math.max(g,b)>20):view.polarity==='light'?(Math.min(r,g,b)>145&&Math.max(r,g,b)-Math.min(r,g,b)<80):Math.max(r,g,b)<90;
            const border=(p/4)%crop.width<3||(p/4)%crop.width>=crop.width-3||Math.floor(p/4/crop.width)<3||Math.floor(p/4/crop.width)>=crop.height-3;
            const tone=view.polarity==='red'?255-Math.min(255,Math.max(0,r-Math.max(g,b))*2.6):view.polarity==='light'?255-Math.min(255,Math.max(0,Math.min(r,g,b)-70)*1.65):(r+g+b)/3;
            soft.data[p]=soft.data[p+1]=soft.data[p+2]=border?255:tone;soft.data[p+3]=255;
            input.data[p]=input.data[p+1]=input.data[p+2]=ink&&!border?0:255;
          }
          softCtx.putImageData(soft,0,0);
          const softReading=await readOCR(softCrop);ocrViews.push({...softReading,turn,region:'isolated-rank-gray'});
          ctx.putImageData(input,0,0);
          const reading=await readOCR(crop);ocrViews.push({...reading,turn,region:'isolated-rank'});
        }
        for(let turn=0;turn<4;turn++){
          const oriented=rotate(flat,turn);
          const crop=canvas(66,60),ctx=crop.getContext('2d');ctx.drawImage(oriented,0,0,66,60,0,0,66,60);
          const input=ctx.getImageData(0,0,66,60);
          const brightness=[];for(let y=0;y<60;y+=5)for(let x=0;x<66;x+=5){const p=(y*66+x)*4;brightness.push((input.data[p]+input.data[p+1]+input.data[p+2])/3);}
          const dark=median(brightness)<135;
          for(let p=0;p<input.data.length;p+=4){
            const r=input.data[p],g=input.data[p+1],b=input.data[p+2];
            const ink=dark?((r>90&&r-g>35&&r-Math.max(g,b)>20)||(Math.min(r,g,b)>145&&Math.max(r,g,b)-Math.min(r,g,b)<80)):(Math.min(r,g,b)<115);
            input.data[p]=input.data[p+1]=input.data[p+2]=ink?0:255;
          }
          ctx.putImageData(input,0,0);
          const reading=await readOCR(crop);
          ocrViews.push({...reading,turn});
        }
        card.ocr=ocrViews;
        const accepted=ocrViews.filter(r=>/^(10|[2-9AJQK])$/.test(r.text)&&r.score>=.90).sort((a,b)=>b.score-a.score);
        const ocr=accepted[0];
        if(ocr){card.rank=ocr.text;card.confidence.ocr=ocr.score;}
        else if(card.confidence.rank<.86)card.rank='';
      }
      if(onProgress)onProgress(i+1,cards.length);
      await new Promise(done=>setTimeout(done,0));
    }
    // Conflicting identities are uncertain, not permission to silently relabel.
    const byCode=new Map();
    for(const card of cards){if(!card.rank||!card.suit)continue;const code=card.rank+card.suit;if(!byCode.has(code))byCode.set(code,[]);byCode.get(code).push(card);}
    for(const same of byCode.values())if(same.length>1)same.forEach(c=>{c.conflict=true;c.rank='';c.suit='';});
    diagnostics.cards=cards.map(c=>({id:c.id,index:c.index,points:c.points,center:c.center,method:c.method,rank:c.rank,suit:c.suit,confidence:c.confidence,ocr:c.ocr,conflict:!!c.conflict,readings:c.readings}));
  }

  window.CardVision={ready,templates,exifOrientation,detect,group,recognize,rotate,rotatePoint,warp};
})();
