window.appearance = (() => {
  let skeletonWidth = 5;
  try {
    const saved = Number(localStorage.getItem('itm-skeleton-width'));
    if (Number.isFinite(saved) && saved >= 1 && saved <= 15) skeletonWidth = saved;
  } catch {}
  const widthInput = document.getElementById('skeleton-width');
  const widthValue = document.getElementById('skeleton-width-value');
  widthInput.value = skeletonWidth;
  widthValue.value = skeletonWidth;
  widthInput.addEventListener('input', () => {
    skeletonWidth = Number(widthInput.value);
    widthValue.value = skeletonWidth;
    try { localStorage.setItem('itm-skeleton-width', String(skeletonWidth)); } catch {}
    window.dispatchEvent(new Event('itm-colors-changed'));
  });
  const groups = [
    ['Motion', [
      ['skeleton-gt', 'GT skeleton', '#247ba0'], ['skeleton-generated', 'Generated skeleton', '#168f79'],
      ['mesh-gt', 'GT SMPL', '#75adc4'], ['mesh-generated', 'Generated SMPL', '#75b8a1'],
      ['canvas', 'Motion background', '#fbfcfc'], ['canvas-text', 'Motion text', '#17202a'],
      ['ground', 'Ground line', '#e4e9eb'],
    ]],
    ['IMU signals', [
      ['imu-x', 'X acceleration', '#d1495b'], ['imu-y', 'Y acceleration', '#2e8b57'],
      ['imu-z', 'Z acceleration', '#247ba0'], ['imu-canvas', 'Signal background', '#ffffff'],
      ['imu-axis', 'Zero line', '#e2e7e9'],
    ]],
    ['Surfaces', [
      ['page', 'Page / toolbar', '#f4f6f7'], ['sidebar', 'Sidebar', '#ffffff'],
      ['card', 'Cards / summaries', '#ffffff'], ['card-header', 'Card headers', '#ffffff'],
      ['header', 'Top bar', '#17202a'], ['empty-cell', 'Empty cells', '#eef2f3'],
      ['border', 'Borders', '#d9dee2'], ['control-border', 'Control borders', '#c8d0d5'],
    ]],
    ['Text', [
      ['text', 'Primary text', '#17202a'], ['muted', 'Secondary text', '#52616b'],
      ['header-text', 'Top bar text', '#ffffff'], ['header-muted', 'Top bar details', '#b8c4c9'],
    ]],
    ['Controls & tags', [
      ['control', 'Buttons / inputs', '#ffffff'], ['control-text', 'Control text', '#17202a'],
      ['accent', 'Accent / sliders', '#168f79'], ['primary-text', 'Primary button text', '#ffffff'],
      ['selected', 'Selected tab', '#edf6f4'], ['tag', 'Condition tags', '#e8f5f2'],
      ['tag-text', 'Condition tag text', '#196b5c'], ['tag-none', 'None tags', '#edf0f2'],
      ['tag-none-text', 'None tag text', '#69767d'], ['status', 'Status background', '#edf6f4'],
      ['status-text', 'Status text', '#31554e'], ['error', 'Error accent', '#d1495b'],
    ]],
  ];
  const defaults = Object.fromEntries(groups.flatMap(([, fields]) => fields.map(([key, , value]) => [key, value])));
  let values = {...defaults};
  try {
    const saved = JSON.parse(localStorage.getItem('itm-colors-v1'));
    for (const key of Object.keys(defaults)) {
      if (/^#[0-9a-f]{6}$/i.test(saved?.[key])) values[key] = saved[key];
    }
  } catch {}
  function apply() {
    for (const [key, value] of Object.entries(values)) document.documentElement.style.setProperty(`--color-${key}`, value);
    window.dispatchEvent(new Event('itm-colors-changed'));
  }
  function save() {
    try { localStorage.setItem('itm-colors-v1', JSON.stringify(values)); } catch {}
  }
  const container = document.getElementById('appearance-fields');
  for (const [name, fields] of groups) {
    const section = document.createElement('fieldset');
    const legend = document.createElement('legend'); legend.textContent = name; section.append(legend);
    for (const [key, title] of fields) {
      const label = document.createElement('label');
      label.textContent = title;
      const input = document.createElement('input');
      input.type = 'color'; input.id = `color-${key}`; input.value = values[key];
      input.title = title;
      input.addEventListener('input', () => { values[key] = input.value; apply(); save(); });
      label.append(input); section.append(label);
    }
    container.append(section);
  }
  document.getElementById('appearance-reset').addEventListener('click', () => {
    values = {...defaults};
    for (const [key, value] of Object.entries(values)) document.getElementById(`color-${key}`).value = value;
    apply(); save();
  });
  apply();
  return {get: key => values[key], skeletonWidth: () => skeletonWidth};
})();
