'use strict';
const $ = s => document.querySelector(s);
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const labels = {success:'成功',failure:'失败',pending:'未评测',running:'执行中',error:'环境异常',unscored:'未评分'};
const stageNames = {initial:'初始观测',direct_straightedge_contact:'直尺接触与推齐',release_and_home:'释放与回位',approach:'接近目标',servo:'末端位置控制',servo_position:'末端位置控制',gripper:'夹爪控制',bound_generated_policy:'绑定本场景生成的策略'};
const stage = s => stageNames[s] || s || '执行动作';
const params = new URLSearchParams(location.search);
const state = {catalog:null,task:params.get('task'),experiment:params.get('run'),seed:'all',status:'all',search:'',selected:params.get('scene'),detail:null,step:Number(params.get('step')||0),tab:'evidence',request:0};
let toastTimer, pollBusy = false;
function toast(message) { $('#toast').textContent=message; $('#toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('#toast').hidden=true,3500); }
async function get(url) { const response = await fetch(url);const data=await response.json();if(!response.ok)throw new Error(data.error||'读取失败');return data; }
function sceneBatch() {return state.catalog.scenes.filter(s=>s.task===state.task && s.experiment===state.experiment);}
function counts(scenes){const c={success:0,failure:0,pending:0,running:0,error:0,unscored:0};scenes.forEach(s=>c[s.status]++);return c;}
function batchName(id){return state.catalog?.experiments.find(e=>e.id===id)?.title||formatBatch(id);}
function formatBatch(name){if(name.endsWith('::self_evolving'))return '无预置 skills · GPT‑6 跨场景自进化';if(name.startsWith('gpt6-skills-')){if(name.endsWith('::with_skills'))return '有 skills · GPT‑6 harness 直控';if(name.endsWith('::without_skills'))return '无 skills · GPT‑6 harness 直控';}if(name.endsWith('::with_skills'))return '有 skills · awomopaw + π0.5';if(name.endsWith('::without_skills'))return '无 skills · awomopaw + π0.5';return name.replace(/-202610\d\d[-T]?/,' · ').replace('physicalrsi-align-all-scenes-local','全场景 · 固定技能').replace('physicalrsi-imported-align-skill','导入技能验证').replace('physicalrsi-skill-seed0-layouts0-4','固定技能 · 5 场景').replace('gpt56-physicalrsi-skill-seed0-layouts0-4','GPT-5.6 · 技能选择');}
function writeLocation(){const q=new URLSearchParams();q.set('task',state.task);q.set('run',state.experiment);if(state.selected)q.set('scene',state.selected);if(state.step)q.set('step',state.step);history.replaceState(null,'','?'+q);}
function updateSelectors(){
  const tasks=state.catalog.tasks;
  if(!tasks.some(t=>t.name===state.task))state.task=tasks.find(t=>t.name==='align_blocks')?.name || tasks[0]?.name;
  $('#task').innerHTML=tasks.map(t=>`<option value="${esc(t.name)}">${esc(t.name)}</option>`).join('');$('#task').value=state.task;
  const available=new Set(state.catalog.scenes.filter(s=>s.task===state.task).map(s=>s.experiment));
  const experiments=state.catalog.experiments.filter(e=>available.has(e.id)).map(e=>e.id);
  if(!experiments.includes(state.experiment))state.experiment=experiments.find(e=>e.endsWith('::with_skills'))||experiments.find(e=>e.startsWith('physicalrsi-align-all-scenes-local'))||experiments.find(e=>e.startsWith('physicalrsi-skill-seed0-layouts0-4'))||experiments.sort((a,b)=>state.catalog.scenes.filter(s=>s.task===state.task&&s.experiment===b).length-state.catalog.scenes.filter(s=>s.task===state.task&&s.experiment===a).length)[0];
  $('#experiment').innerHTML=experiments.map(e=>`<option value="${esc(e)}">${esc(batchName(e))}</option>`).join('');$('#experiment').value=state.experiment;
  const seeds=[...new Set(sceneBatch().map(s=>s.seed))].sort((a,b)=>a-b);
  if(state.seed!=='all'&&!seeds.some(s=>String(s)===state.seed))state.seed='all';
  $('#seed').innerHTML='<option value="all">全部 seed</option>'+seeds.map(s=>`<option value="${esc(s)}">Seed ${esc(s)}</option>`).join('');$('#seed').value=state.seed;
}
function visibleScenes(){return sceneBatch().filter(s=>{
  if(state.seed!=='all'&&String(s.seed)!==state.seed)return false;
  if(state.status==='other'&&['success','failure'].includes(s.status))return false;
  if(!['all','other'].includes(state.status)&&s.status!==state.status)return false;
  if(state.search){const q=state.search.trim();const parts=q.split('-');if(parts.length===2)return Number(parts[0])===Number(s.seed)&&Number(parts[1])===Number(s.layout);return !isNaN(Number(q))?Number(s.layout)===Number(q):String(s.run_id||'').includes(q);}
  return true;
});}
function renderScenes(){
  const batch=sceneBatch(),c=counts(batch),valid=c.success+c.failure;
  const online=batch.some(s=>s.online_code_evolution);
  $('#batch-summary').innerHTML=`<div class="rate">${valid?(100*c.success/valid).toFixed(online?2:1):'—'}<span style="font-size:14px">${valid?'%':''}</span><small>${batch.some(s=>s.paired_validation)?'候选验证成功率（含冷启动）':online?'逐场更新 · 在线成功率':'已评分成功率'}</small></div><p>${valid} / ${batch.length} 已评分 · ${c.success} 成功 · ${c.failure} 失败</p><div class="progress"><span style="width:${batch.length?100*valid/batch.length:0}%"></span></div>`;
  $('#filters button[data-status="success"]').textContent=`成功 ${c.success}`;$('#filters button[data-status="failure"]').textContent=`失败 ${c.failure}`;
  $('#filters').querySelectorAll('button').forEach(b=>b.classList.toggle('active',b.dataset.status===state.status));
  const groups=new Map();visibleScenes().sort((a,b)=>(a.seed-b.seed)||(a.layout-b.layout)).forEach(s=>{if(!groups.has(s.seed))groups.set(s.seed,[]);groups.get(s.seed).push(s);});
  $('#scene-list').innerHTML=groups.size?[...groups].map(([seed,scenes])=>{
    const all=batch.filter(s=>s.seed===seed),n=counts(all);
    return `<div class="seed-heading"><strong>Seed ${esc(seed)}</strong><span>${n.success} / ${n.success+n.failure} 成功 · ${all.length} 场景</span></div><div class="seed-grid">${scenes.map(s=>`<button class="scene ${esc(s.status)} ${s.id===state.selected?'selected':''}" data-scene="${s.id}" aria-label="Seed ${esc(s.seed)} 场景 ${esc(s.layout)} ${labels[s.status]}" aria-pressed="${s.id===state.selected}" title="${esc(labels[s.status]+' · '+s.provider+' · '+s.variant)}">${esc(String(s.layout).padStart(3,'0'))}</button>`).join('')}</div>`;
  }).join(''):'<div class="empty-note">没有符合筛选条件的场景。</div>';
}
async function chooseVisible(){
  const visible=visibleScenes();
  if(!visible.some(s=>s.id===state.selected)){state.selected=visible.find(s=>s.status==='success')?.id||visible[0]?.id;state.step=0;}
  renderScenes();
  if(state.selected)await selectScene(state.selected);else{$('#workspace').innerHTML='<div class="empty-state">没有符合筛选条件的场景，试试调整筛选。</div>';state.detail=null;}
}
function badge(s){return `<span class="badge ${esc(s.status)}">${labels[s.status]||'未知'}</span>`;}
function camera(name,label,sub,main=false){
  const url=state.detail.cameras[name];
  return `<div class="camera ${main?'main-camera':''}">${url?`<video ${main?'id="main-video"':'data-wrist="true"'} src="${esc(url)}" preload="auto" playsinline muted aria-label="${label}"></video>`:`<div class="camera-empty"><span>▧</span>暂无${sub}视频<small>${state.detail.status==='pending'?'此场景尚未评测':'该次执行未保存可用视频'}</small></div>`}<div class="camera-label">${label}<small>${sub}</small></div>${url?`<a class="expand" href="${esc(url)}" target="_blank" rel="noopener" aria-label="单独打开${sub}视频" style="display:grid;place-items:center">↗</a>`:''}${main?'<div id="frame-caption" class="frame-caption">初始观测</div>':''}</div>`;
}
function renderDetail(){
  const d=state.detail;const pair=d.comparison;const paired=pair?`<div class="instruction"><div><strong>下一场景配对验证</strong><p>${pair.status==='bootstrap'?'场景 0：空 skill、policy、memory 的观测初始化。':`旧版 ${esc(pair.incumbent||'待执行')}：${pair.incumbent_score==null?'—':esc(pair.incumbent_score)} · 候选 ${esc(pair.candidate||'待生成')}：${pair.candidate_score==null?'—':esc(pair.candidate_score)}<br>${pair.decision==='inherit_child'?'原选择器采用候选':pair.decision==='retain_parent'?'原选择器保留旧版':'等待原选择器比较'}${pair.survivor?' · '+esc(pair.survivor):''}`}</p><small>主视频和成功率展示候选的原始执行结果；旧版视频及完整选择记录见原始产物。</small></div></div>`:'';const isAgent=d.provider==='gpt6-harness'||d.messages.some(m=>m.role==='assistant');
  $('#workspace').innerHTML=`<div class="breadcrumb"><span>${esc(d.task)}</span><span>/</span><span>${esc(batchName(d.experiment))}</span></div>
  <div class="workspace-heading"><div><div class="title-line"><h2>Episode ${esc(String(d.layout).padStart(3,'0'))}</h2>${badge(d)}</div><div class="episode-meta">SEED ${esc(d.seed)}${d.framework?' &nbsp; / &nbsp; 框架 '+esc(d.framework):''} &nbsp; / &nbsp; ${esc(d.provider)}${d.arm?' &nbsp; / &nbsp; '+esc(d.arm==='self_evolving'?'跨场景自进化':d.arm==='with_skills'?'有 skills':'无 skills'):''}</div></div><div class="metrics"><div class="metric"><span>官方评分</span><strong>${d.score==null?'—':esc(d.score)}</strong></div><div class="metric"><span>动作步</span><strong>${d.total_steps||'—'}</strong></div><div class="metric"><span>相机视角</span><strong>${Object.keys(d.cameras).length} / 3</strong></div></div></div>
  ${d.error?`<div class="error-box">${esc(d.error)}</div>`:''}
  ${d.qualification?`<div class="instruction">${esc(d.qualification)}</div>`:''}
  ${paired}
  <div class="layout"><section class="viewer"><div class="instruction"><span class="symbol">▸</span><div><strong>任务指令</strong> &nbsp; ${esc(d.instruction || (d.status==='pending'?'等待执行后记录官方任务指令。':'此记录未保存官方任务指令。'))}</div></div>
  ${camera('cam_head','HEAD CAMERA','头部视角',true)}
  <div class="playback"><div class="transport"><button id="prev-step" title="前一步" aria-label="前一步">‹</button><button id="play" title="播放 / 暂停" aria-label="播放 / 暂停">▶</button><button id="next-step" title="后一步" aria-label="后一步">›</button><input id="scrub" type="range" min="0" max="${d.total_steps||0}" value="${state.step}" aria-label="动作步进度"><span id="timecode" class="timecode">00:00 / 00:00</span><select id="speed" aria-label="播放速度"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></div><div class="playback-note"><span>三路视角同步播放</span><span>视频与动作按帧序近似对齐 · ← → 步进</span></div></div>
  <div class="secondary-heading"><span>WRIST CAMERAS / 腕部视角</span><span>与主视角同步</span></div><div class="camera-grid">${camera('cam_left_wrist','LEFT WRIST','左腕')}${camera('cam_right_wrist','RIGHT WRIST','右腕')}</div><div class="viewer-footer">RUN ${esc(d.run_id||'尚未执行')}<br>轨迹来源：${esc(d.trajectory_source)}</div></section>
  <aside class="trace-column"><section class="card"><div class="card-head"><b>01</b> ${isAgent?'AGENT / 决策记录':'SKILL / 执行入口'}</div><h3>${d.online_code_evolution?'本场景执行的策略':isAgent?'Agent 选择与计划':'已选择任务技能'}</h3><div class="skill-name">${esc(d.policy_id||d.provider)}<span class="badge">${d.online_code_evolution?'Code-policy':isAgent?'Agent 调用':'评测配置'}</span></div><p>${d.video_rethink_enabled?'策略由上一场景的执行反馈及失败视频复盘生成，再在本场景执行；首场景从空经验起步。':d.online_code_evolution?'模型根据当前观测和已完成场景的反馈生成本场景策略。下方为保存的策略修改理由。':isAgent?'下方展示这次执行保存的可见决策摘要与工具调用。':'通过评测配置指定 provider，执行其返回的动作。'}</p>${d.rationale?`<details><summary>查看原始执行计划</summary><pre>${esc(d.rationale)}</pre></details>`:''}<div class="source-note">本集：<strong>${esc(d.task)}</strong><br>${isAgent?`${d.messages.filter(m=>m.role==='assistant').length} 条可见消息 · ${d.calls.length} 次工具调用`:'此批次未保存 Agent 可见决策记录'}</div></section>
  <section class="card execution-card"><div class="card-head"><b>02</b> EXECUTION / 当前步骤</div><h3 id="step-title">初始观测</h3><div class="step-counter"><span>动作步</span><strong id="step-counter">0 / ${d.total_steps}</strong></div><div id="joints"></div><div class="source-note">关节数值为 harness 执行的目标命令，单位 rad。</div></section>
  <section class="card"><div class="card-head"><b>03</b> TRACE / 轨迹证据</div><div class="tabs"><button data-tab="evidence">中间结果</button><button data-tab="agent">Agent</button><button data-tab="raw">原始日志</button></div><div id="log-content" class="log-box"></div></section>
  <section class="card"><div class="card-head"><b>04</b> ARTIFACTS / 原始产物</div>${d.files.map(f=>`<a class="file-link" href="${esc(f.url)}" target="_blank" rel="noopener">${esc(f.label)}<span>↗</span></a>`).join('')||'<div class="empty-note">尚无执行产物。</div>'}</section></aside></div>`;
  const videos=[...$('#workspace').querySelectorAll('video')],main=$('#main-video');
  $('#play').disabled=!main;$('#speed').disabled=!main;
  $('#scrub').disabled=!d.total_steps;$('#prev-step').disabled=!d.total_steps;$('#next-step').disabled=!d.total_steps;
  $('#play').onclick=async()=>{if(!main)return;try{if(main.paused){if(main.ended)seekStep(0);await Promise.all(videos.map(v=>v.play()));}else videos.forEach(v=>v.pause());}catch{toast('视频暂时无法播放，可用右上角链接单独打开。');}};
  $('#scrub').oninput=e=>seekStep(Number(e.target.value));
  $('#prev-step').onclick=()=>seekStep(state.step-1);$('#next-step').onclick=()=>seekStep(state.step+1);
  $('#speed').onchange=e=>videos.forEach(v=>v.playbackRate=Number(e.target.value));
  $('#workspace').querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{state.tab=b.dataset.tab;renderLog();});
  videos.forEach(v=>v.addEventListener('error',()=>toast('视频解码失败，请单独打开原始视频查看。')));
  videos.filter(v=>v!==main).forEach(v=>v.addEventListener('loadedmetadata',()=>{if(main&&Number.isFinite(main.duration))v.currentTime=Math.min(main.currentTime,v.duration);}));
  if(main){
    main.addEventListener('loadedmetadata',()=>{seekStep(state.step,false);updateTimecode();});
    main.addEventListener('play',()=>$('#play').textContent='Ⅱ');main.addEventListener('pause',()=>{$('#play').textContent='▶';videos.forEach(v=>{if(v!==main)v.pause();});});
    main.addEventListener('ended',()=>{$('#play').textContent='▶';videos.forEach(v=>v.pause());});
    main.addEventListener('timeupdate',()=>{
      const next=videoStep(main);if(next!==state.step){state.step=next;renderStep();}
      updateTimecode();videos.filter(v=>v!==main).forEach(v=>{if(Math.abs(v.currentTime-main.currentTime)>.15&&Number.isFinite(v.duration))v.currentTime=Math.min(main.currentTime,v.duration);});
    });
  }
  renderStep();renderLog();
}
function videoStep(main){const frames=state.detail.video_frames||state.detail.total_steps+2;return Math.max(0,Math.min(state.detail.total_steps,Math.floor(main.currentTime/main.duration*frames)||0));}
function timeText(t){if(!Number.isFinite(t))return '00:00';return String(Math.floor(t/60)).padStart(2,'0')+':'+String(Math.floor(t%60)).padStart(2,'0');}
function updateTimecode(){const main=$('#main-video');if(main&&$('#timecode'))$('#timecode').textContent=timeText(main.currentTime)+' / '+timeText(main.duration);}
function seekStep(value,pause=true){
  if(!state.detail)return;state.step=Math.max(0,Math.min(state.detail.total_steps,Math.round(value)));
  const main=$('#main-video');const frames=state.detail.video_frames||state.detail.total_steps+2;
  if(main&&Number.isFinite(main.duration)){
    const time=state.step===state.detail.total_steps?Math.max(0,main.duration-.03):state.step/frames*main.duration;
    $('#workspace').querySelectorAll('video').forEach(v=>{if(pause)v.pause();if(Number.isFinite(v.duration))v.currentTime=Math.min(time,Math.max(0,v.duration-.03));});
  }
  renderStep();writeLocation();
}
function renderStep(){
  const d=state.detail;if(!d||!$('#step-title'))return;
  const s=d.steps.filter(x=>x.step<=state.step).at(-1),action=s?.action;
  const title=state.step===0?'初始观测':stage(s?.stage);
  $('#step-title').textContent=title;$('#step-counter').textContent=state.step+' / '+d.total_steps;$('#scrub').value=state.step;
  $('#frame-caption').textContent=state.step===0?'初始观测':'第 '+state.step+' 步 · '+title;
  const residual=d.evidence.filter(e=>e.dls_residual!=null&&e.count<=state.step).at(-1);
  $('#joints').innerHTML=(residual?`<div class="residual"><span>最近记录的 IK 残差</span><strong>${(residual.dls_residual*1000).toFixed(2)} mm</strong></div>`:'')+(action?`<table><thead><tr><th>目标关节</th><th>左臂 / rad</th><th>右臂 / rad</th></tr></thead><tbody>${Array.from({length:Math.max(action.left_arm_joint_state?.length||0,action.right_arm_joint_state?.length||0)},(_,i)=>`<tr><td>J${i+1}</td><td>${num(action.left_arm_joint_state?.[i])}</td><td>${num(action.right_arm_joint_state?.[i])}</td></tr>`).join('')}<tr><td>夹爪命令</td><td>${num(action.left_ee_joint_state?.[0],2)}</td><td>${num(action.right_ee_joint_state?.[0],2)}</td></tr></tbody></table>`:`<div class="empty-note">${state.step===0?'选择动作步以查看执行记录。':'该步没有保存关节目标。'}${s?.execution?`<br>本次执行 ${esc(s.execution.executed)} 个动作` :''}</div>`);
  $('#log-content')?.querySelectorAll('[data-evidence-step]').forEach(b=>b.parentElement.classList.toggle('active',Number(b.dataset.evidenceStep)<=state.step));
}
function num(v,digits=4){return typeof v==='number'?v.toFixed(digits):'—';}
function evidenceTitle(e){return e.contact?'计算接触点与推行目标':e.dls_residual!=null?'逆运动学目标':stage(e.stage);}
function evidenceText(e){if(e.contact)return '接触点 ['+e.contact.map(v=>num(v,3)).join(', ')+'] m';if(e.target)return '目标 ['+e.target.map(v=>num(v,3)).join(', ')+'] m'+(e.dls_residual!=null?' · 残差 '+num(e.dls_residual*1000,2)+' mm':'');return JSON.stringify(e).slice(0,200);}
function renderLog(){
  const d=state.detail;if(!d)return;
  $('#workspace').querySelectorAll('[data-tab]').forEach(b=>b.classList.toggle('active',b.dataset.tab===state.tab));
  const content=$('#log-content');
  if(state.tab==='evidence'){
    const evidence=d.evidence.length?d.evidence:d.events.filter(e=>e.event==='action_chunk_ready').filter((e,i,a)=>!i||e.subtask_goal!==a[i-1].subtask_goal||e.provider_id!==a[i-1].provider_id).map(e=>({count:d.events.filter(x=>x.event==='action_chunk_execution'&&x.line<e.line).reduce((n,x)=>n+(x.executed||0),0),stage:e.subtask_goal,line:e.line,planning_summary:e.planning_summary}));
    content.innerHTML=evidence.length?`<ol class="evidence-list">${evidence.map(e=>`<li><button data-evidence-step="${e.count||0}"><small>第 ${e.count||0} 步 · 原日志行 ${e.line}</small><strong>${esc(evidenceTitle(e))}</strong><p>${esc(e.planning_summary||evidenceText(e))}</p></button></li>`).join('')}</ol>`:'<div class="empty-note">此记录没有保存中间结果。可查看 Agent 记录或原始日志。</div>';
    if(d.learning){
      const l=d.learning,v=l.video_rethinking;
      const video=v?`<strong>VideoRethink · 视频复盘</strong><p>${v.status==='completed'?`${v.cameras} 路视频 · ${v.frames} 个抽帧 · 诊断与 rethinking 已完成`:v.status==='bootstrap'?'空经验观测初始化，按原流程不诊断任务策略。':v.status==='not_required'?'本场景成功，按原流程不触发失败视频复盘。':'尚未生成本场景的失败视频复盘。'}</p>${v.diagnosis?`<p>${esc(v.diagnosis)}</p>`:''}${v.policy_changes.length?`<p style="white-space:pre-wrap">${esc(v.policy_changes.join('\n'))}</p>`:''}`:'';
      content.insertAdjacentHTML('afterbegin',`<div class="log-event">${video}<strong>跨场景经验 v${esc(l.version)}</strong><p>${l.kind==='code-policy'?'累计反馈来源场景':'来源失败场景'}：${esc(l.source_layouts.join(', ')||'暂无，从空经验开始')}</p><p style="white-space:pre-wrap">${esc(l.task_memory||'本场景未使用预置技能或历史经验。')}</p>${l.failure_analysis.failure_summary?`<strong>本场景失败分析</strong><p>${esc(l.failure_analysis.failure_summary)}</p>`:''}${l.update_rationale?`<strong>本场景反馈后的策略更新</strong><p>${esc(l.update_rationale)}</p>`:''}</div>`);
    }
    content.querySelectorAll('[data-evidence-step]').forEach(b=>b.onclick=()=>seekStep(Number(b.dataset.evidenceStep)));
  }else if(state.tab==='agent'){
    content.innerHTML=d.messages.map(m=>`<div class="log-event"><strong>${esc(m.role?.toUpperCase())}</strong><p style="white-space:pre-wrap">${esc(m.text)}</p></div>`).join('')+d.calls.map(c=>`<details class="log-event"><summary>${esc(c.name)}</summary><pre>${esc(JSON.stringify(c.input,null,2))}</pre></details>`).join('');
    if(!content.innerHTML)content.innerHTML='<div class="empty-note">此批次没有保存 Agent 的可见摘要或工具调用记录。</div>';
  }else{
    content.innerHTML=d.events.length?d.events.map(e=>`<details class="log-event"><summary>行 ${e.line} · ${esc(e.event)}</summary><pre>${esc(JSON.stringify(e,null,2))}</pre></details>`).join(''):'<div class="empty-note">没有 Harness 事件日志，可从原始产物打开其他记录。</div>';
  }
  renderStep();
}
async function selectScene(id){
  const request=++state.request;
  if(id!==state.selected){state.step=0;state.selected=id;}
  renderScenes();
  try{const detail=await get('/api/scenes/'+id);if(request!==state.request)return;$('#workspace').querySelectorAll('video').forEach(v=>v.pause());state.detail=detail;state.step=Math.min(state.step,detail.total_steps);renderDetail();writeLocation();}
  catch(error){if(request===state.request){$('#workspace').innerHTML='<div class="empty-state">'+esc(error.message)+'</div>';toast(error.message);}}
}
async function refresh(manual=false){
  if(pollBusy)return;pollBusy=true;
  try{
    state.catalog=await get('/api/catalog'+(manual?'?refresh=1':''));
    if(!state.catalog.scenes.length){$('#workspace').innerHTML='<div class="empty-state">尚无可识别的评测记录。</div>';return;}
    updateSelectors();renderScenes();
    $('#sync-label').textContent='本地数据 · '+new Date().toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit'});
    if(!state.detail||manual||!sceneBatch().some(s=>s.id===state.selected))await chooseVisible();
    else{const updated=state.catalog.scenes.find(s=>s.id===state.selected);if(updated&&(updated.status!==state.detail.status||updated.status==='running'||updated.evidence_revision!==state.detail.evidence_revision))await selectScene(updated.id);}
    if(manual)toast('评测记录已更新');
  }catch(error){toast(error.message);if(!state.detail)$('#workspace').innerHTML='<div class="empty-state">无法读取评测记录：'+esc(error.message)+'</div>';}
  finally{pollBusy=false;}
}
$('#task').onchange=async e=>{state.task=e.target.value;state.experiment=null;state.selected=null;state.seed='all';updateSelectors();await chooseVisible();};
$('#experiment').onchange=async e=>{state.experiment=e.target.value;state.selected=null;state.seed='all';updateSelectors();await chooseVisible();};
$('#seed').onchange=async e=>{state.seed=e.target.value;await chooseVisible();};
$('#filters').onclick=async e=>{const b=e.target.closest('[data-status]');if(b){state.status=b.dataset.status;await chooseVisible();}};
$('#search').oninput=e=>{state.search=e.target.value;renderScenes();};
$('#scene-list').onclick=e=>{const b=e.target.closest('[data-scene]');if(b)selectScene(b.dataset.scene);};
$('#refresh').onclick=()=>refresh(true);
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA','BUTTON','SUMMARY'].includes(e.target.tagName)||e.altKey||e.ctrlKey||e.metaKey||!state.detail)return;if(e.key==='ArrowLeft'){e.preventDefault();seekStep(state.step-1);}if(e.key==='ArrowRight'){e.preventDefault();seekStep(state.step+1);}if(e.code==='Space'){e.preventDefault();$('#play')?.click();}});
refresh();setInterval(()=>{if(!document.hidden)refresh();},15000);
