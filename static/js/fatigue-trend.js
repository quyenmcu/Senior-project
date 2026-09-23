(() => {
  const COLORS = { grid: "#dbe5f0", text: "#61758b" };

  function draw(canvas) {
    const source = document.getElementById(canvas.dataset.source);
    if (!source) return;
    const data = JSON.parse(source.textContent);
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    const ctx = canvas.getContext("2d");
    ctx.scale(ratio, ratio);
    const pad = { left: 42, right: 18, top: 18, bottom: 42 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    const maximum = Math.max(1, ...data.series.flatMap((line) => line.values));
    const yMax = Math.max(5, Math.ceil(maximum / 5) * 5);

    ctx.font = "11px Manrope, sans-serif";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i += 1) {
      const y = pad.top + (plotH * i) / 5;
      ctx.strokeStyle = COLORS.grid;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(width - pad.right, y); ctx.stroke();
      ctx.fillStyle = COLORS.text;
      ctx.textAlign = "right";
      ctx.fillText(String(Math.round(yMax * (1 - i / 5))), pad.left - 8, y + 4);
    }
    const x = (index) => pad.left + (plotW * index) / Math.max(1, data.labels.length - 1);
    const y = (value) => pad.top + plotH - (value / yMax) * plotH;
    const labelStep = Math.max(1, Math.ceil(data.labels.length / 6));
    data.labels.forEach((label, index) => {
      if (index % labelStep !== 0 && index !== data.labels.length - 1) return;
      ctx.fillStyle = COLORS.text; ctx.textAlign = "center";
      ctx.fillText(label.slice(5), x(index), height - 14);
    });
    data.series.forEach((line) => {
      ctx.strokeStyle = line.color; ctx.fillStyle = line.color; ctx.lineWidth = 2.5;
      ctx.beginPath();
      line.values.forEach((value, index) => index ? ctx.lineTo(x(index), y(value)) : ctx.moveTo(x(index), y(value)));
      ctx.stroke();
      line.values.forEach((value, index) => { ctx.beginPath(); ctx.arc(x(index), y(value), 3, 0, Math.PI * 2); ctx.fill(); });
    });
    canvas._trend = { data, pad, plotW, width };
  }

  function showTooltip(canvas, event) {
    const state = canvas._trend;
    if (!state) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const index = Math.max(0, Math.min(state.data.labels.length - 1,
      Math.round(((mouseX - state.pad.left) / state.plotW) * (state.data.labels.length - 1))));
    const tip = canvas.parentElement.querySelector(".trend-tooltip");
    tip.innerHTML = `<strong>${state.data.labels[index]}</strong>${state.data.series.map(line => `<span><i style="background:${line.color}"></i>${line.name}: ${line.values[index]}</span>`).join("")}`;
    tip.hidden = false;
    tip.style.left = `${Math.min(state.width - 145, Math.max(8, mouseX + 10))}px`;
    tip.style.top = `${Math.max(8, event.clientY - rect.top - 80)}px`;
  }

  const charts = [...document.querySelectorAll("canvas.fatigue-trend")];
  const redraw = () => charts.forEach(draw);
  charts.forEach((canvas) => {
    canvas.addEventListener("mousemove", (event) => showTooltip(canvas, event));
    canvas.addEventListener("mouseleave", () => { canvas.parentElement.querySelector(".trend-tooltip").hidden = true; });
  });
  window.addEventListener("resize", redraw);
  redraw();
})();
