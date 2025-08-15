(async function() {
  const statusEl = document.getElementById('status');
  try {
    const r = await fetch('/health');
    const j = await r.json();
    statusEl.textContent = j.status === 'ok' ? 'Healthy' : 'Degraded';
    if (j.status === 'ok') statusEl.style.color = 'var(--mint)';
  } catch (e) {
    statusEl.textContent = 'Offline';
    statusEl.style.color = 'var(--solar-orange)';
  }
})();


