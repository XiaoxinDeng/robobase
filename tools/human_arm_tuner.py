#!/usr/bin/env python3
from __future__ import annotations
import argparse, base64, io, json, os, queue, shlex, signal, subprocess, sys, threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PYTHON = Path('/home/xd1125/miniconda3/envs/safe_bigym_hoi/bin/python')
SNAPSHOT = 'exp_local/pixel_act/bigym_drawer_top_open_20260528034109/snapshots/3000_snapshot.pt'
MANIFEST = '/home/xd1125/.bigym/demonstrations/0.9.0/DrawerTopOpen/JointPositionActionMode_floating_pelvis_x_pelvis_y_pelvis_z_pelvis_rz_absolute/lightweight/manifest.json'

HTML = r'''
<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Human Arm Tuner</title>
<style>
:root{color-scheme:dark;--bg:#111318;--panel:#1b1f27;--line:#343b49;--text:#e8ebf1;--muted:#a5adba;--accent:#5db7ff}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}header{display:flex;justify-content:space-between;align-items:center;padding:14px 18px;border-bottom:1px solid var(--line);background:#151922;position:sticky;top:0}h1{font-size:16px;margin:0}main{display:grid;grid-template-columns:360px 1fr;gap:14px;padding:14px}section{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}h2{font-size:13px;margin:0;padding:10px 12px;border-bottom:1px solid var(--line);background:#171c25}.controls{display:grid;gap:12px;padding:12px}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:8px}label{display:flex;justify-content:space-between;color:var(--muted);font-size:12px}input{width:100%}input[type=range]{accent-color:var(--accent)}input[type=text],input[type=number]{background:#222734;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:6px}button{background:#273143;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:8px 10px;font-weight:650;cursor:pointer}.primary{background:#1d5f91}.danger{background:#5f2931}.buttons{display:flex;gap:8px;flex-wrap:wrap}canvas,.preview{width:100%;max-width:760px;aspect-ratio:1.45;border:1px solid var(--line);border-radius:8px;background:#0f1218;object-fit:contain}pre{margin:0;padding:12px;background:#0d1016;white-space:pre-wrap;overflow:auto;max-height:360px}.log{height:330px}.pill{border:1px solid var(--line);border-radius:999px;padding:4px 8px;color:var(--muted);background:#1d2330}.small{color:var(--muted);font-size:12px;padding:0 12px 12px}@media(max-width:900px){main{grid-template-columns:1fr}}
</style></head><body>
<header><h1>Human Arm Mirror Tuner</h1><span class="pill" id="state">idle</span></header>
<main><section><h2>Temporary Blocker</h2><div class="controls">
    <div><label>Checkpoint</label><input id="snapshot" type="text"></div>
    <div class="grid2"><div><label>Trigger dist <span id="triggerDistv"></span></label><input id="triggerDist" type="range" min="0.05" max="1.20" step="0.01"></div><div><label>Natural motion <span id="naturalScalev"></span></label><input id="naturalScale" type="range" min="0" max="2.5" step="0.05"></div></div>
    <div class="grid2"><div><label>Enter sec <span id="enterDurationv"></span></label><input id="enterDuration" type="range" min="0.2" max="3.0" step="0.05"></div><div><label>Hold sec <span id="holdDurationv"></span></label><input id="holdDuration" type="range" min="0" max="3.0" step="0.05"></div></div>
    <div><label>Exit sec <span id="exitDurationv"></span></label><input id="exitDuration" type="range" min="0.2" max="3.0" step="0.05"></div>
    <div class="grid2"><div><label>Spawn X <span id="spawnXv"></span></label><input id="spawnX" type="range" min="-1.40" max="0.40" step="0.01"></div><div><label>Spawn Y <span id="spawnYv"></span></label><input id="spawnY" type="range" min="-0.40" max="1.00" step="0.01"></div></div>
    <div class="grid2"><div><label>Move X <span id="moveXv"></span></label><input id="moveX" type="range" min="-1.80" max="1.80" step="0.01"></div><div><label>Move Y <span id="moveYv"></span></label><input id="moveY" type="range" min="0.00" max="1.20" step="0.01"></div></div>
    <div class="grid2"><div><label>Block yaw <span id="blockYawv"></span></label><input id="blockYaw" type="range" min="-1.57" max="1.57" step="0.01"></div><div><label>Block pitch <span id="blockPitchv"></span></label><input id="blockPitch" type="range" min="-1.40" max="0.40" step="0.01"></div></div>
    <div><label>Block elbow <span id="blockElbowv"></span></label><input id="blockElbow" type="range" min="0.10" max="2.40" step="0.01"></div>
    <div><label>Preview step <span id="previewStepv"></span></label><input id="previewStep" type="range" min="0" max="180" step="1"></div>
    <div class="grid2"><div><label>Steps</label><input id="steps" type="number" min="20" max="3500" step="10"></div><div><label>Video steps</label><input id="videoSteps" type="number" min="20" max="3500" step="10"></div></div>
    <div><label>Run name</label><input id="runName" type="text"></div><div class="buttons"><button class="primary" id="preview">Render MuJoCo preview</button><button class="primary" id="run">Run smoke</button><button class="danger" id="stop">Stop</button><button id="copy">Copy command</button><button id="reset">Reset</button></div>
    </div><div class="small">Built-in TemporaryDrawerArmBlocker tuning. Preview force-triggers the blocker immediately; smoke runs the clean drawer task with a mirrored human safety/video env.</div></section>
<div style="display:grid;gap:14px"><section><h2>MuJoCo Preview</h2><div style="padding:12px"><img class="preview" id="mj" alt="MuJoCo preview"><div class="small" id="previewStatus">Click Render MuJoCo preview. Use Preview step to inspect before/after release.</div></div></section><section><h2>Carrier Path Sketch</h2><div style="padding:12px"><canvas id="view" width="900" height="620"></canvas></div></section><section><h2>Command</h2><pre id="cmd"></pre></section><section><h2>Log</h2><pre class="log" id="log"></pre></section><section><h2>Latest result</h2><pre id="result">No run yet.</pre></section></div></main>
<script>
const SNAPSHOT='exp_local/pixel_act/bigym_drawer_top_open_20260528034109/snapshots/3000_snapshot.pt';
const MANIFEST='/home/xd1125/.bigym/demonstrations/0.9.0/DrawerTopOpen/JointPositionActionMode_floating_pelvis_x_pelvis_y_pelvis_z_pelvis_rz_absolute/lightweight/manifest.json';
const defaults={snapshot:SNAPSHOT,triggerDist:1.20,enterDuration:1.20,holdDuration:1.20,exitDuration:1.20,naturalScale:1.0,spawnX:0.06,spawnY:-0.31,moveX:0.00,moveY:0.62,blockYaw:1.57,blockPitch:-0.45,blockElbow:1.00,previewStep:40,steps:3500,videoSteps:260,runName:"temporary_blocker_mirror_tune"};
const ids=Object.keys(defaults), el=id=>document.getElementById(id); let evt=null;
function vals(){let v={};ids.forEach(k=>{let n=el(k);v[k]=n.type==='text'?n.value:Number(n.value)});return v} function setVals(v){ids.forEach(k=>el(k).value=v[k]);update()} function fmt(x){return Number(x).toFixed(3).replace(/0+$/,'').replace(/\.$/,'')}
function blockXY(v){return [v.spawnX+v.moveX,v.spawnY+v.moveY]}
function qOutside(v){return [v.spawnX,v.spawnY,0,0,-0.35,1.20]}
    function qBlock(v){let b=blockXY(v);return [b[0],b[1],0,v.blockYaw,v.blockPitch,v.blockElbow]}
    function qList(q){return '['+q.map(fmt).join(',')+']'}
    function command(v){return `/home/xd1125/miniconda3/envs/safe_bigym_hoi/bin/python eval_act_oscbf_safety_metrics.py \
  --condition act --snapshot ${v.snapshot} --env bigym/drawer_top_open --safety-env bigym/human_arm_drawer_top_open --normalization-source eval \
  --episodes 1 --steps ${v.steps} --demos 40 --out ${v.runName}.jsonl --output-dir eval_safety/${v.runName} \
  --video-time-base sim --stop-video-at-steps ${v.videoSteps} --continue-after-success \
  --override env.episode_length=400000 --override +env.enable_temporary_human_blocker=true --override +env.trigger_dist=${fmt(v.triggerDist)} \
  --override +env.enter_duration=${fmt(v.enterDuration)} --override +env.hold_duration=${fmt(v.holdDuration)} --override +env.exit_duration=${fmt(v.exitDuration)} \
  --override +env.natural_motion_scale=${fmt(v.naturalScale)} --override +env.q_outside=${qList(qOutside(v))} --override +env.q_block=${qList(qBlock(v))} --override +env.q_exit=null \
  --override frame_stack=4 --override execution_length=1`}
    function wc(x,y){let c=el('view'),minX=-1.25,maxX=.95,minY=-.45,maxY=1.05;return[(x-minX)/(maxX-minX)*c.width,c.height-(y-minY)/(maxY-minY)*c.height]}
    function draw(){let c=el("view"),g=c.getContext("2d"),v=vals(),bxy=blockXY(v);g.clearRect(0,0,c.width,c.height);g.fillStyle="#0f1218";g.fillRect(0,0,c.width,c.height);g.strokeStyle="#263140";g.lineWidth=1;for(let x=-1.2;x<=.9;x+=.2){let a=wc(x,-.45),b=wc(x,1.05);g.beginPath();g.moveTo(a[0],a[1]);g.lineTo(b[0],b[1]);g.stroke()}for(let y=-.4;y<=1.0;y+=.2){let a=wc(-1.25,y),b=wc(.95,y);g.beginPath();g.moveTo(a[0],a[1]);g.lineTo(b[0],b[1]);g.stroke()}let robot=wc(0,0),handle=wc(.55,.15),outside=wc(v.spawnX,v.spawnY),block=wc(bxy[0],bxy[1]);g.strokeStyle="#5db7ff";g.lineWidth=4;g.beginPath();g.moveTo(robot[0],robot[1]);g.lineTo(handle[0],handle[1]);g.stroke();g.fillStyle="#5db7ff";g.beginPath();g.arc(robot[0],robot[1],9,0,7);g.fill();g.fillStyle="#d9edff";g.fillText("robot to handle path",robot[0]+12,robot[1]-12);g.fillStyle="#9ccfff";g.fillRect(handle[0]-36,handle[1]-14,72,28);g.fillStyle="#d9edff";g.fillText("drawer handle",handle[0]-34,handle[1]+34);g.strokeStyle="#f0a44d";g.lineWidth=3;g.beginPath();g.moveTo(outside[0],outside[1]);g.lineTo(block[0],block[1]);g.stroke();g.fillStyle="#7bd88f";g.beginPath();g.arc(outside[0],outside[1],9,0,7);g.fill();g.fillText("spawn",outside[0]+12,outside[1]-10);g.fillStyle="#f0a44d";g.beginPath();g.arc(block[0],block[1],10,0,7);g.fill();g.fillText("block",block[0]+12,block[1]-10);g.strokeStyle="rgba(240,164,77,.45)";g.beginPath();g.arc(handle[0],handle[1],v.triggerDist*260,0,7);g.stroke()}
    function update(){let v=vals();["triggerDist","enterDuration","holdDuration","exitDuration","naturalScale","spawnX","spawnY","moveX","moveY","blockYaw","blockPitch","blockElbow","previewStep"].forEach(k=>{let o=el(k+"v");if(o)o.textContent=fmt(v[k])});draw();el("cmd").textContent=command(v)} ids.forEach(k=>el(k).addEventListener("input",update));el("reset").onclick=()=>setVals(defaults);el("copy").onclick=()=>navigator.clipboard.writeText(el("cmd").textContent);el("run").onclick=async()=>{el("log").textContent="Starting...\n";el("result").textContent="Running...";await fetch("/api/run",{method:"POST",body:JSON.stringify(vals())});connect()};el("stop").onclick=async()=>fetch("/api/stop",{method:"POST"});el("preview").onclick=async()=>{let v=vals();el("previewStatus").textContent="Rendering MuJoCo preview at step "+v.previewStep+"...";let r=await fetch("/api/preview",{method:"POST",body:JSON.stringify(v)});let d=await r.json();if(d.image){el("mj").src=d.image;el("previewStatus").textContent=d.message||"Rendered."}else{el("previewStatus").textContent=d.error||"Preview failed."}}
function connect(){if(evt)evt.close();evt=new EventSource('/api/events');evt.onmessage=e=>{let d=JSON.parse(e.data);el('state').textContent=d.running?'running':'idle';if(d.line){el('log').textContent+=d.line;el('log').scrollTop=el('log').scrollHeight}if(d.result)show(d.result)}} function show(r){let txt=JSON.stringify(r.summary||{},null,2);if(r.video)txt='video: '+r.video+'\nsummary: '+r.summary_path+'\n\n'+txt;el('result').textContent=txt} async function poll(){let d=await (await fetch('/api/status')).json();el('state').textContent=d.running?'running':'idle';if(d.result)show(d.result)} setVals(defaults);poll();setInterval(poll,3000);
</script></body></html>
'''

state = {'proc': None, 'lines': queue.Queue(), 'result': None}
preview_env = None
preview_lock = threading.Lock()

def clean_name(name):
    s = ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in str(name).strip())
    return s or 'tune_drawer_local'

def _fmt_float(value):
    return (f"{float(value):.6f}").rstrip('0').rstrip('.')

def _q_outside(v):
    return [float(v.get("spawnX", v.get("outsideX", 0.06))), float(v.get("spawnY", v.get("outsideY", -0.31))), 0.0, 0.0, -0.35, 1.20]

def _q_block(v):
    outside = _q_outside(v)
    if "moveX" in v or "moveY" in v:
        block_x = outside[0] + float(v.get("moveX", 0.00))
        block_y = outside[1] + float(v.get("moveY", 0.62))
    else:
        block_x = float(v.get("blockX", 0.06))
        block_y = float(v.get("blockY", 0.25))
    return [block_x, block_y, 0.0, float(v.get("blockYaw", 1.57)), float(v.get("blockPitch", -0.45)), float(v.get("blockElbow", 1.0))]

def _list_override(values):
    return '[' + ','.join(_fmt_float(x) for x in values) + ']'

def blocker_overrides(v, *, preview=False, add_keys=False):
    trigger = 10.0 if preview else float(v.get('triggerDist', 1.20))
    prefix = '+env.' if add_keys else 'env.'
    return [
        prefix + 'enable_temporary_human_blocker=true',
        prefix + 'trigger_dist=' + _fmt_float(trigger),
        prefix + 'enter_duration=' + _fmt_float(v.get('enterDuration', 1.2)),
        prefix + 'hold_duration=' + _fmt_float(v.get('holdDuration', 1.2)),
        prefix + 'exit_duration=' + _fmt_float(v.get('exitDuration', 1.2)),
        prefix + 'natural_motion_scale=' + _fmt_float(v.get('naturalScale', 1.0)),
        prefix + 'q_outside=' + _list_override(_q_outside(v)),
        prefix + 'q_block=' + _list_override(_q_block(v)),
        prefix + 'q_exit=null',
    ]
def build_cmd(v):
    name = clean_name(v.get('runName', 'temporary_blocker_tune'))
    steps = int(v.get('steps', 260)); vsteps = int(v.get('videoSteps', steps))
    snapshot = str(v.get('snapshot') or SNAPSHOT)
    cmd = [str(PYTHON), 'eval_act_oscbf_safety_metrics.py', '--condition', 'act', '--snapshot', snapshot, '--env', 'bigym/drawer_top_open', '--safety-env', 'bigym/human_arm_drawer_top_open', '--normalization-source', 'eval', '--episodes', '1', '--steps', str(steps), '--demos', '40', '--out', name + '.jsonl', '--output-dir', 'eval_safety/' + name, '--video-time-base', 'sim', '--stop-video-at-steps', str(vsteps), '--continue-after-success', '--override', 'env.episode_length=400000']
    for override in blocker_overrides(v, add_keys=True):
        cmd.extend(['--override', override])
    cmd.extend(['--override', 'frame_stack=4', '--override', 'execution_length=1'])
    return cmd

def get_preview_env(v):
    global preview_env
    from robobase.safetyfilter.eval_utils.eval_utils import make_cfg, make_eval_env
    if preview_env is not None:
        try: preview_env.close()
        except Exception: pass
    overrides = blocker_overrides(v, preview=True) + ['env.manifest=' + MANIFEST, 'env.privileged_information=false', 'env.require_mode_label=false', 'frame_stack=4', 'execution_length=1']
    args = SimpleNamespace(env='bigym/human_arm_drawer_top_open', demos=1, episodes=1, override=overrides)
    preview_env = make_eval_env(make_cfg(args))
    return preview_env

def render_preview(v):
    import numpy as np
    from PIL import Image
    with preview_lock:
        env = get_preview_env(v)
        env.reset(seed=0)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        for _ in range(max(0, int(v.get('previewStep', 0))) + 1):
            env.step(action)
        frame = env.render()
        bio = io.BytesIO(); Image.fromarray(frame).save(bio, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(bio.getvalue()).decode('ascii')

def result_for(name):
    root = ROOT / 'eval_safety' / name
    sp = root / (name + '_summary.json'); vp = root / 'videos' / 'act_episode_000.mp4'
    summary = None
    if sp.exists():
        try: summary = json.loads(sp.read_text())
        except Exception as exc: summary = {'error': str(exc)}
    return {'run_name': name, 'summary': summary, 'summary_path': str(sp.relative_to(ROOT)) if sp.exists() else None, 'video': str(vp.relative_to(ROOT)) if vp.exists() else None}

def reader(proc, name):
    assert proc.stdout
    for line in proc.stdout: state['lines'].put(line)
    proc.wait(); state['lines'].put('\n[process exited: %s]\n' % proc.returncode); state['result'] = result_for(name); state['proc'] = None

def start(v):
    if state.get('proc') is not None: return False, 'run already active'
    name = clean_name(v.get('runName', 'tune_drawer_local')); cmd = build_cmd(v); state['result'] = None
    while not state['lines'].empty():
        try: state['lines'].get_nowait()
        except queue.Empty: break
    state['lines'].put('[command]\n' + ' '.join(shlex.quote(x) for x in cmd) + '\n\n')
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
    state['proc'] = proc; threading.Thread(target=reader, args=(proc, name), daemon=True).start(); return True, name

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def send_json(self, obj, status=HTTPStatus.OK):
        data = json.dumps(obj).encode('utf-8'); self.send_response(status); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)
    def read_body(self):
        n = int(self.headers.get('Content-Length', '0')); return json.loads(self.rfile.read(n).decode('utf-8')) if n else {}
    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/':
            data = HTML.encode('utf-8'); self.send_response(200); self.send_header('Content-Type', 'text/html; charset=utf-8'); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data); return
        if path == '/api/status': self.send_json({'running': state.get('proc') is not None, 'result': state.get('result')}); return
        if path == '/api/events':
            self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.send_header('Cache-Control', 'no-cache'); self.end_headers(); last = None
            for _ in range(3600):
                payload = {'running': state.get('proc') is not None, 'line': ''}
                try: payload['line'] = state['lines'].get(timeout=1)
                except queue.Empty: pass
                if state.get('result') is not None and state.get('result') is not last: payload['result'] = state.get('result'); last = state.get('result')
                try: self.wfile.write(('data: ' + json.dumps(payload) + '\n\n').encode('utf-8')); self.wfile.flush()
                except BrokenPipeError: break
            return
        self.send_error(404)
    def do_POST(self):
        path = urlparse(self.path).path
        if path == '/api/run': ok, msg = start(self.read_body()); self.send_json({'ok': ok, 'message': msg}, 200 if ok else 409); return
        if path == '/api/preview':
            try: self.send_json({'image': render_preview(self.read_body()), 'message': 'Rendered from MuJoCo.'})
            except Exception as exc: self.send_json({'error': repr(exc)}, 500)
            return
        if path == '/api/stop':
            proc = state.get('proc')
            if proc:
                try: os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                state['lines'].put('\n[stop requested]\n')
            self.send_json({'ok': True}); return
        self.send_error(404)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--host', default='127.0.0.1'); parser.add_argument('--port', type=int, default=8765); args = parser.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'Human arm tuner running at http://{args.host}:{args.port}', flush=True)
    print('In VSCode Remote, forward this port from the Ports panel.', flush=True)
    httpd.serve_forever()
if __name__ == '__main__': main()
