(() => {
  const canvas = document.getElementById("premium-series-chart");
  const source = document.getElementById("premium-series-data");
  if (!canvas || !source) return;
  const data = JSON.parse(source.textContent);

  function draw() {
    const actual = data.history;
    const projected = data.forecast;
    const points = [...actual, ...projected];
    if (!points.length) return;
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    canvas.width = width * ratio; canvas.height = height * ratio;
    const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio);
    const pad = { left: 68, right: 22, top: 20, bottom: 46 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    const values = points.map((point) => point.amount);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const spread = Math.max(50, max - min);
    const yMin = Math.max(0, Math.floor((min - spread * .2) / 50) * 50);
    const yMax = Math.ceil((max + spread * .2) / 50) * 50;
    const x = (index) => pad.left + plotW * index / Math.max(1, points.length - 1);
    const y = (value) => pad.top + plotH - ((value - yMin) / Math.max(1, yMax - yMin)) * plotH;
    ctx.font = "11px Manrope, sans-serif";
    for (let i = 0; i <= 5; i += 1) {
      const lineY = pad.top + plotH * i / 5;
      ctx.strokeStyle = "#dbe5f0"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(pad.left, lineY); ctx.lineTo(width - pad.right, lineY); ctx.stroke();
      ctx.fillStyle = "#61758b"; ctx.textAlign = "right";
      ctx.fillText(`NT$ ${Math.round(yMax - (yMax - yMin) * i / 5)}`, pad.left - 8, lineY + 4);
    }
    const step = Math.max(1, Math.ceil(points.length / 7));
    points.forEach((point, index) => {
      if (index % step && index !== points.length - 1) return;
      ctx.fillStyle = "#61758b"; ctx.textAlign = "center";
      ctx.fillText(point.date.slice(5), x(index), height - 15);
    });
    const drawLine = (series, offset, color, dashed) => {
      if (!series.length) return;
      ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 3;
      ctx.setLineDash(dashed ? [8, 6] : []); ctx.beginPath();
      series.forEach((point, index) => {
        const globalIndex = offset + index;
        if (index) ctx.lineTo(x(globalIndex), y(point.amount));
        else ctx.moveTo(x(globalIndex), y(point.amount));
      }); ctx.stroke(); ctx.setLineDash([]);
      series.forEach((point, index) => { ctx.beginPath(); ctx.arc(x(offset + index), y(point.amount), 4, 0, Math.PI * 2); ctx.fill(); });
    };
    drawLine(actual, 0, "#1677e8", false);
    const bridge = actual.length ? [actual[actual.length - 1], ...projected] : projected;
    drawLine(bridge, Math.max(0, actual.length - 1), "#8b5cf6", true);
    canvas._points = { points, x, width };
  }

  canvas.addEventListener("mousemove", (event) => {
    const state = canvas._points; if (!state) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    let best = 0;
    state.points.forEach((_, index) => { if (Math.abs(state.x(index) - mouseX) < Math.abs(state.x(best) - mouseX)) best = index; });
    const point = state.points[best];
    const tip = canvas.parentElement.querySelector(".premium-tooltip");
    tip.innerHTML = `<strong>${point.date}</strong><span>NT$ ${Number(point.amount).toFixed(2)}</span>`;
    tip.hidden = false; tip.style.left = `${Math.min(state.width - 135, Math.max(8, mouseX + 10))}px`;
    tip.style.top = `${Math.max(8, event.clientY - rect.top - 55)}px`;
  });
  canvas.addEventListener("mouseleave", () => { canvas.parentElement.querySelector(".premium-tooltip").hidden = true; });
  document.querySelectorAll(".forecast-tab").forEach((button) => button.addEventListener("click", () => {
    document.querySelectorAll(".forecast-tab").forEach((item) => item.classList.toggle("active", item === button));
    document.querySelectorAll(".forecast-list").forEach((list) => { list.hidden = list.id !== button.dataset.target; });
  }));
  window.addEventListener("resize", draw); draw();
})();
