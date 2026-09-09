/* Three.js r160.1, vendored locally; SMPL surfaces are fitted preview assets. */
window.meshPreview = (() => {
  let renderer, scene, camera, pitchGroup, yawGroup;
  let current = null;
  let running = false;
  let pending = false;
  const mode = () => document.getElementById('mesh-mode').value;
  const status = text => document.getElementById('mesh-status').textContent = text;
  function initialize() {
    if (renderer) return;
    const next = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    next.setSize(480, 355);
    next.setClearColor(appearance.get('canvas'));
    scene = new THREE.Scene();
    scene.add(new THREE.HemisphereLight(0xffffff, 0x71828a, 2));
    const light = new THREE.DirectionalLight(0xffffff, 2);
    light.position.set(2, 4, 5); scene.add(light);
    camera = new THREE.OrthographicCamera(-480/210, 480/210, 315/105, -40/105, .1, 100);
    camera.position.z = 10;
    pitchGroup = new THREE.Group(); yawGroup = new THREE.Group();
    pitchGroup.add(yawGroup); scene.add(pitchGroup);
    renderer = next;
  }
  function reset(result) {
    if (current) current.panels.forEach(panel => {
      panel.meshPreview?.geometry.dispose(); panel.meshPreview?.material.dispose();
      delete panel.meshPreview;
    });
    current = result;
    document.getElementById('mesh-clear').disabled = running || !current;
    status('');
    document.getElementById('mesh-retry').hidden = true;
    if (mode() !== 'skeleton') prepare();
  }
  async function prepare() {
    if (running) { pending = true; return; }
    if (!current || mode() === 'skeleton') return;
    running = true;
    document.getElementById('mesh-clear').disabled = true;
    const result = current;
    try {
      initialize();
      let workerCount = 2;
      if (document.getElementById('mesh-device').value === 'gpu_all') {
        const response = await fetch('/api/mesh-devices');
        if (!response.ok) throw new Error('Unable to discover fitting GPUs');
        const devices = await response.json();
        workerCount = Math.max(1, devices.gpus.length);
      }
      let nextIndex = 0;
      const worker = async () => { while (nextIndex < result.panels.length) {
        const i = nextIndex++;
        if (current !== result || mode() === 'skeleton') return;
        if (result.panels[i].meshPreview) continue;
        let response, data;
        for (let attempt = 0; attempt < 300; attempt++) {
          if (current !== result || mode() === 'skeleton') return;
          status(`SMPL ${result.panels.filter(p => p.meshPreview).length}/${result.panels.length} ready; fitting / queued...`);
          response = await fetch('/api/mesh', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({
            ...source(result), panel_index:i, device:document.getElementById('mesh-device').value,
          })});
          data = await response.json();
          if (response.status !== 409) break;
          await new Promise(resolve => setTimeout(resolve, 2000));
        }
        if (!response.ok) throw new Error(data.detail || 'SMPL fitting failed');
        if (current !== result) return;
        const responses = await Promise.all([fetch(data.vertices_url, {cache:'no-store'}), fetch(data.faces_url, {cache:'no-store'})]);
        if (responses.some(value => !value.ok)) throw new Error('Failed to load SMPL cache');
        const [buffer, faces] = await Promise.all([responses[0].arrayBuffer(), responses[1].json()]);
        if (current !== result) return;
        if (buffer.byteLength !== data.frames * data.vertices * 3 * 4) throw new Error('Invalid mesh cache size');
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(data.vertices * 3), 3));
        geometry.setIndex(faces);
        const material = new THREE.MeshStandardMaterial({color:appearance.get(result.panels[i].kind === 'ground_truth' ? 'mesh-gt' : 'mesh-generated'), roughness:.8, side:THREE.DoubleSide});
        const mesh = new THREE.Mesh(geometry, material); mesh.frustumCulled = false;
        result.panels[i].meshPreview = {geometry, material, mesh, data, vertices:new Float32Array(buffer), frame:-1};
        const tag = document.createElement('span'); tag.className = 'tag smpl-fit-tag';
        tag.textContent = `SMPL fit ${(data.joint_error_m * 100).toFixed(1)} cm`;
        tag.title = `Mean fitted joint distance; ${data.device || 'cpu'}; ${data.fit_seconds?.toFixed(1) || '?'} s`;
        document.getElementById(`motion-${i}`)?.parentElement.querySelector('.tags')?.append(tag);
        renderFrame();
      }};
      const completed = await Promise.allSettled(Array.from({length:Math.min(workerCount, result.panels.length)}, () => worker()));
      const failed = completed.find(value => value.status === 'rejected');
      if (failed) throw failed.reason;
      if (current === result) status(mode() === 'skeleton' ? '' : 'Fitted SMPL preview');
    } catch (error) {
      if (current === result) {
        status(error.message); document.getElementById('mesh-retry').hidden = false;
      }
    } finally {
      running = false;
      document.getElementById('mesh-clear').disabled = !current;
      if (pending) { pending = false; prepare(); }
    }
  }
  function source(result) {
    return {run_id:result.source?.relative_path || result.run_id, root:result.source?.root_key || null};
  }
  document.getElementById('mesh-clear').addEventListener('click', async () => {
    if (!current || running) return;
    const result = current;
    document.getElementById('mesh-clear').disabled = true;
    try {
      const response = await fetch('/api/mesh/clear', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(source(result))});
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Cache clear failed');
      if (current !== result) return;
      document.getElementById('mesh-mode').value = 'skeleton';
      document.querySelectorAll('.smpl-fit-tag').forEach(tag => tag.remove());
      reset(result); renderFrame();
      status(`Cleared ${data.removed} mesh caches. Identical motions share this cache.`);
    } catch (error) { status(error.message); }
    finally { document.getElementById('mesh-clear').disabled = running || !current; }
  });
  function draw(view, frame) {
    const value = view.panel.meshPreview;
    if (!value || mode() === 'skeleton') return false;
    const index = Math.min(frame, value.data.frames - 1);
    if (value.frame !== index) {
      const stride = value.data.vertices * 3;
      value.geometry.attributes.position.array.set(value.vertices.subarray(index * stride, (index + 1) * stride));
      value.geometry.attributes.position.needsUpdate = true;
      value.geometry.computeVertexNormals(); value.frame = index;
    }
    const root = view.panel.motion[index][0];
    value.material.color.set(appearance.get(view.panel.kind === 'ground_truth' ? 'mesh-gt' : 'mesh-generated'));
    renderer.setClearColor(appearance.get('canvas'));
    value.mesh.position.set(-root[0], 0, -root[2]);
    yawGroup.clear(); yawGroup.add(value.mesh);
    yawGroup.rotation.y = -view.yaw; pitchGroup.rotation.x = view.pitch;
    renderer.render(scene, camera);
    view.canvas.getContext('2d').drawImage(renderer.domElement, 0, 0);
    return mode() === 'mesh';
  }
  document.getElementById('mesh-mode').addEventListener('change', () => { prepare(); renderFrame(); });
  document.getElementById('mesh-device').addEventListener('change', () => prepare());
  document.getElementById('mesh-retry').addEventListener('click', () => {
    document.getElementById('mesh-retry').hidden = true; prepare();
  });
  return {reset, draw};
})();
