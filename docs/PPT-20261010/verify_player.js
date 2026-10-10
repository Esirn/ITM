// CPU smoke check of the standalone HTML's real drawing/control code.
const fs=require('fs'), vm=require('vm');
const html=fs.readFileSync(process.argv[2],'utf8');
const data=html.match(/<script type="application\/json" id="data">([\s\S]*?)<\/script>/)[1];
const js=html.match(/<script>([\s\S]*?)<\/script>/)[1];
let draws=0;
const coordinate=(...args)=>{if(args.some(x=>typeof x==='number'&&!Number.isFinite(x)))throw Error('Non-finite drawing coordinate');};
const ctx={clearRect(){draws++;},fillText(){},beginPath(){},stroke(){},fill(){},moveTo:coordinate,lineTo:coordinate,arc:coordinate};
const elements={data:{textContent:data},case:{value:0,appendChild(){}},canvas:{width:1380,height:620,getContext(){return ctx;}},time:{},caption:{},clock:{},note:{},play:{},mode:{}};
const sandbox={document:{getElementById:id=>elements[id],createElement:()=>({})},requestAnimationFrame(){}};
vm.createContext(sandbox);vm.runInContext(js,sandbox,{timeout:30000});
vm.runInContext(`for(let i=0;i<cases.length;i++){select.value=i;select.onchange();slider.value=cases[i].frames-1;slider.oninput();document.getElementById('mode').onclick();document.getElementById('mode').onclick();document.getElementById('play').onclick();}`,sandbox,{timeout:30000});
process.stdout.write(JSON.stringify({cases:6,draw_calls:draws,controls:'case selection, full-range slider, world toggle, play',finite_coordinates:true,actual_browser_rendering:false}));
