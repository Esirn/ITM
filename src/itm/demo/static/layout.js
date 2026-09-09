(() => {
  const container = document.getElementById('panels');
  const settings = document.getElementById('layout-settings');
  let original = null;
  let cards = [];
  let positions = [];
  let grid = null;
  let selected = null;
  let key = '';
  let rows = 4;
  let cols = 3;
  const status = message => document.getElementById('layout-status').textContent = message;
  const save = () => {
    try { localStorage.setItem(key, JSON.stringify({ rows, cols, positions })); }
    catch { status('Layout applied; browser storage unavailable.'); }
  };
  function move(index, destination) {
    const other = positions.indexOf(destination);
    if (other !== -1) positions[other] = positions[index];
    positions[index] = destination;
    selected = null;
    place();
    save();
  }
  function place() {
    cards.forEach((card, index) => grid.children[positions[index]].append(card));
    [...grid.children].forEach((cell, index) => {
      cell.classList.toggle('occupied', positions.includes(index));
      cell.classList.remove('drop-target');
    });
  }
  function build() {
    if (!original) {
      original = document.createDocumentFragment();
      while (container.firstChild) original.append(container.firstChild);
      cards.forEach(card => {
        const marker = document.createComment('layout-card');
        card.before(marker);
        card.layoutMarker = marker;
      });
    }
    if (grid) grid.remove();
    grid = document.createElement('section');
    grid.className = 'custom-layout';
    grid.style.gridTemplateColumns = `repeat(${cols}, var(--panel-size))`;
    for (let i = 0; i < rows * cols; i++) {
      const cell = document.createElement('div');
      cell.className = 'layout-cell';
      cell.dataset.position = `${Math.floor(i / cols) + 1}, ${i % cols + 1}`;
      cell.tabIndex = 0;
      cell.setAttribute('aria-label', `Row ${Math.floor(i / cols) + 1}, column ${i % cols + 1}`);
      cell.addEventListener('dragover', event => {
        if (selected === null || !settings.open) return;
        event.preventDefault(); cell.classList.add('drop-target');
      });
      cell.addEventListener('dragleave', () => cell.classList.remove('drop-target'));
      cell.addEventListener('drop', event => {
        event.preventDefault(); if (selected !== null) move(selected, i);
      });
      const activate = () => { if (settings.open && selected !== null) move(selected, i); };
      cell.addEventListener('click', activate);
      cell.addEventListener('keydown', event => {
        if (event.target === cell && ['Enter', ' '].includes(event.key)) { event.preventDefault(); activate(); }
      });
      grid.append(cell);
    }
    container.append(grid);
    cards.forEach((card, index) => {
      if (card.querySelector('.layout-handle')) return;
      const handle = document.createElement('button');
      handle.className = 'layout-handle';
      handle.textContent = '\u2807';
      handle.title = 'Move card: drag, or select then choose a cell';
      handle.setAttribute('aria-label', `Move ${card.querySelector('h3')?.textContent || 'card'}`);
      handle.draggable = true;
      handle.addEventListener('dragstart', event => {
        selected = index; event.dataTransfer.setData('text/plain', String(index));
        event.dataTransfer.effectAllowed = 'move';
      });
      handle.addEventListener('dragend', () => { selected = null; grid.querySelectorAll('.drop-target').forEach(cell => cell.classList.remove('drop-target')); });
      handle.addEventListener('click', event => { event.stopPropagation(); selected = index; status('Select a destination cell.'); });
      card.prepend(handle);
    });
    place();
    container.classList.toggle('layout-editing', settings.open);
  }
  function restore() {
    if (!original) return;
    cards.forEach(card => { card.querySelector('.layout-handle')?.remove(); card.layoutMarker.replaceWith(card); });
    container.replaceChildren(original);
    original = null; grid = null; selected = null;
    container.classList.remove('layout-editing');
  }
  document.getElementById('layout-apply').addEventListener('click', () => {
    if (!cards.length) return status('Generate or load a result first.');
    const r = Number(document.getElementById('layout-rows').value);
    const c = Number(document.getElementById('layout-cols').value);
    if (!Number.isInteger(r) || !Number.isInteger(c) || r < 1 || r > 100 || c < 1 || c > 12 || r * c < cards.length) {
      return status(`Use 1–100 rows, 1–12 columns and at least ${cards.length} cells.`);
    }
    rows = r; cols = c;
    if (positions.some(position => position >= rows * cols)) positions = cards.map((_, index) => index);
    build(); save(); status(`${rows} × ${cols} · ${cards.length} cards`);
  });
  document.getElementById('layout-reset').addEventListener('click', () => {
    restore(); try { localStorage.removeItem(key); } catch {}
    positions = cards.map((_, index) => index); status('Default layout restored.');
  });
  settings.addEventListener('toggle', () => {
    selected = null; container.classList.toggle('layout-editing', settings.open);
  });
  new MutationObserver(() => {
    // Result rendering replaces direct children; moving cards within our grid does not.
    if (grid && grid.parentNode === container) return;
    const next = [...container.querySelectorAll('.motion-panel')];
    if (!next.length || next[0] === cards[0]) return;
    original = null; grid = null; selected = null; cards = next;
    positions = cards.map((_, index) => index);
    key = `itm-layout-v1:${JSON.stringify(cards.map(card => card.querySelector('h3').textContent))}`;
    rows = Math.ceil(cards.length / 3); cols = 3;
    try {
      const saved = JSON.parse(localStorage.getItem(key));
      if (saved && Number.isInteger(saved.rows) && saved.rows > 0 && saved.rows <= 100 && Number.isInteger(saved.cols) && saved.cols > 0 && saved.cols <= 12 && Array.isArray(saved.positions) && saved.positions.length === cards.length && new Set(saved.positions).size === cards.length && saved.positions.every(p => Number.isInteger(p) && p >= 0 && p < saved.rows * saved.cols)) {
        ({ rows, cols, positions } = saved); build();
      }
    } catch {}
    document.getElementById('layout-rows').value = rows;
    document.getElementById('layout-cols').value = cols;
    status(`${cards.length} cards`);
  }).observe(container, { childList: true });
})();
