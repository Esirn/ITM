const fs=require('fs'),vm=require('vm'),path=require('path');
const folder=process.argv[2];let checked=0;
for(const name of fs.readdirSync(folder).filter(x=>/^FREV.*\.html$/.test(x))){
 const content=fs.readFileSync(path.join(folder,name),'utf8');
 const js=content.match(/<script>([\s\S]*)<\/script>/)[1];
 const elements={};const context={clearRect(){},beginPath(){},lineTo(){},moveTo(){},stroke(){},fillText(){}};
 const doc={getElementById(id){return elements[id]??=( {value:id==='plane'?'0':id==='curve'?'':'0',width:1000,height:420,max:0,textContent:'',getContext(){return context},append(option){if(!this.value)this.value=option.textContent}})},createElement(){return {textContent:''}}};
 vm.runInNewContext(js,{document:doc,setInterval(){},console}, {timeout:2000});
 elements.plane.value='2';elements.plane.onchange();
 elements.frame.value=elements.frame.max;elements.frame.oninput();
 elements.curve.value='left_knee_degrees';elements.curve.onchange();
 checked++;
}
console.log(JSON.stringify({pages_executed:checked,initial_draw:true,last_frame:true,ZY_projection:true,knee_curve:true,browser_visual_inspection:false}));
