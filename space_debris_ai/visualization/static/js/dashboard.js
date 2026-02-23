// A.R.I.E.S Dashboard JavaScript
// AetherOS-style interactive visualizations

const COLORS = {
    bg: '#050508',
    panel: 'rgba(12, 14, 24, 0.85)',
    cyan: '#00d4ff',
    blue: '#58a6ff',
    orange: '#ff8c42',
    green: '#22c55e',
    purple: '#a78bfa',
    textPrimary: '#e2e8f0',
    textSecondary: '#64748b',
    grid: 'rgba(100, 116, 139, 0.15)',
};

const EARTH_RADIUS = 6371.0; // km

// Common layout settings for Plotly
const commonLayout = {
    paper_bgcolor: COLORS.panel,
    plot_bgcolor: COLORS.panel,
    font: {
        family: 'Courier New, monospace',
        color: COLORS.textPrimary,
        size: 11,
    },
    margin: { t: 30, r: 30, b: 50, l: 60 },
    showlegend: true,
    legend: {
        bgcolor: COLORS.panel,
        bordercolor: COLORS.grid,
        borderwidth: 1,
        font: { size: 10 },
    },
};

// Last loaded mission data (for resize redraw without refetch)
let lastMissionData = null;

/** Generate a seed in [0, 2e9] from time (ms) and random component. */
const MAX_SEED = 2000000000;
function generateSeed() {
    const timePart = Date.now() % (MAX_SEED + 1);
    const randomPart = Math.floor(Math.random() * 1000);
    return (timePart + randomPart) % (MAX_SEED + 1);
}

/** Fetch mission data with the given seed and render the dashboard. */
async function loadMissionData(seed) {
    const url = '/api/mission-data' + (seed != null ? '?seed=' + encodeURIComponent(seed) : '');
    // #region agent log
    fetch('http://127.0.0.1:7552/ingest/ded2ad03-5f07-437e-b0e7-36793f4588e3',{method:'POST',headers:{'Content-Type':'application/json','X-Debug-Session-Id':'17e329'},body:JSON.stringify({sessionId:'17e329',location:'dashboard.js:loadMissionData',message:'loadMissionData_called',data:{seed:seed,url:url},timestamp:Date.now(),hypothesisId:'H4'})}).catch(function(){});
    // #endregion
    const response = await fetch(url);
    // #region agent log
    fetch('http://127.0.0.1:7552/ingest/ded2ad03-5f07-437e-b0e7-36793f4588e3',{method:'POST',headers:{'Content-Type':'application/json','X-Debug-Session-Id':'17e329'},body:JSON.stringify({sessionId:'17e329',location:'dashboard.js:fetch_done',message:'loadMissionData_fetch_done',data:{ok:response.ok,status:response.status},timestamp:Date.now(),hypothesisId:'H4'})}).catch(function(){});
    // #endregion
    if (!response.ok) throw new Error('Mission data request failed');
    const data = await response.json();
    lastMissionData = data;

    renderOrbitChart(data);
    renderPositionChart(data);
    renderVelocityChart(data);
    renderResourcesChart(data);
    updateVelocityGauge(data);
    updateReadouts(data);
    updateTrajectoryFooter(data);
    drawRadar(data);
    updateHealthBars(data);
    updateCopilotCards(data);
    updateThreatCount(data);
    updateBatteryShields(data);
    updateDangerStatus(data);
    updateDangerWarnings(data);

    if (data.seed != null) {
        const seedEl = document.getElementById('seed-value');
        const blockEl = document.getElementById('current-seed');
        if (seedEl) seedEl.textContent = data.seed;
        if (blockEl) blockEl.style.display = 'inline';
    }
    return data;
}

function startLiveTime() {
    const el = document.getElementById('live-time');
    if (!el) return;
    function tick() {
        const d = new Date();
        el.textContent = d.toTimeString().slice(0, 8);
    }
    tick();
    setInterval(tick, 1000);
}

function updateVelocityGauge(data) {
    const speeds = data.speeds || [];
    const speedKmh = speeds.length ? speeds[speeds.length - 1] * 3600 : 0;
    const gaugeValue = document.getElementById('gauge-value');
    const gaugeArc = document.getElementById('gauge-arc');
    if (gaugeValue) gaugeValue.textContent = speedKmh.toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
    if (gaugeArc) {
        const maxKmh = 40000;
        const ratio = Math.min(1, speedKmh / maxKmh);
        gaugeArc.style.strokeDashoffset = 245 * (1 - ratio);
    }
}

function updateReadouts(data) {
    const s = data.spacecraft_status || {};
    const altEl = document.getElementById('readout-alt');
    if (altEl && s.altitude != null) altEl.textContent = s.altitude.toFixed(1) + ' km';
}

function updateTrajectoryFooter(data) {
    const times = data.times || [];
    const positions = data.positions || [];
    const etaEl = document.getElementById('eta');
    const distEl = document.getElementById('dist');
    const headingEl = document.getElementById('heading');
    if (etaEl && times.length) etaEl.textContent = (times[times.length - 1] / 60).toFixed(2) + ' min ETA';
    if (distEl && positions.length >= 2) {
        const p0 = positions[0], p1 = positions[positions.length - 1];
        const d = Math.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2 + (p1[2]-p0[2])**2);
        distEl.textContent = d.toFixed(0) + ' km DIST';
    }
    if (headingEl) headingEl.textContent = 'NNE 024° HEADING';
}

function drawRadar(data) {
    const canvas = document.getElementById('radar-canvas');
    const debrisPos = data.debris_positions || [];
    const positions = data.positions || [];
    if (!canvas || !canvas.getContext) return;

    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    const w = canvas.width = rect.width || canvas.offsetWidth || 218;
    const h = canvas.height = rect.height || canvas.offsetHeight || 218;
    const cx = w / 2, cy = h / 2, r = Math.min(w, h) / 2 - 8;

    ctx.clearRect(0, 0, w, h);

    // Concentric circles
    ctx.strokeStyle = 'rgba(0, 212, 255, 0.2)';
    ctx.lineWidth = 1;
    for (let i = 1; i <= 4; i++) {
        ctx.beginPath();
        ctx.arc(cx, cy, r * i / 4, 0, Math.PI * 2);
        ctx.stroke();
    }
    ctx.strokeStyle = 'rgba(0, 212, 255, 0.4)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();

    // Crosshairs
    ctx.strokeStyle = 'rgba(0, 212, 255, 0.15)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, cy - r);
    ctx.lineTo(cx, cy + r);
    ctx.moveTo(cx - r, cy);
    ctx.lineTo(cx + r, cy);
    ctx.stroke();

    const scPos = positions.length ? positions[positions.length - 1] : [0, 0, 0];

    // Spacecraft at center
    ctx.fillStyle = COLORS.cyan;
    ctx.strokeStyle = COLORS.cyan;
    ctx.shadowColor = COLORS.cyan;
    ctx.shadowBlur = 8;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.shadowBlur = 0;

    const safeDebris = [];
    const dangerousDebris = [];
    debrisPos.forEach(p => {
        const dx = p[0] - scPos[0], dy = p[1] - scPos[1], dz = (p[2] || 0) - (scPos[2] || 0);
        const dist = Math.sqrt(dx*dx + dy*dy + dz*dz) || 1;
        if (dist < 0.1) dangerousDebris.push({pos: p, dist: dist, dx: dx, dy: dy});
        else safeDebris.push({pos: p, dist: dist, dx: dx, dy: dy});
    });

    safeDebris.forEach(d => {
        const dist2d = Math.sqrt(d.dx*d.dx + d.dy*d.dy) || 1;
        const maxRange = 500;
        const scale = Math.min(1, maxRange / d.dist);
        const x = cx + (d.dx / dist2d) * r * scale * 0.75;
        const y = cy - (d.dy / dist2d) * r * scale * 0.75;
        ctx.fillStyle = COLORS.green;
        ctx.shadowColor = COLORS.green;
        ctx.shadowBlur = 6;
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
    });

    dangerousDebris.forEach(d => {
        const dist2d = Math.sqrt(d.dx*d.dx + d.dy*d.dy) || 1;
        const maxRange = 500;
        const scale = Math.min(1, maxRange / d.dist);
        const x = cx + (d.dx / dist2d) * r * scale * 0.75;
        const y = cy - (d.dy / dist2d) * r * scale * 0.75;
        ctx.fillStyle = '#ff4444';
        ctx.strokeStyle = '#ff4444';
        ctx.shadowColor = '#ff4444';
        ctx.shadowBlur = 12;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x - 6, y - 6);
        ctx.lineTo(x + 6, y + 6);
        ctx.moveTo(x + 6, y - 6);
        ctx.lineTo(x - 6, y + 6);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, Math.PI * 2);
        ctx.stroke();
        ctx.shadowBlur = 0;
    });
}

function updateHealthBars(data) {
    const s = data.spacecraft_status || {};
    const fuels = data.fuels || [];
    const conf = data.confidences || [];
    const fuelPct = fuels.length ? (fuels[fuels.length - 1] / (fuels[0] || 1)) * 100 : 100;
    const fusionPct = conf.length ? conf[conf.length - 1] * 100 : 100;
    const setBar = (id, pct) => {
        const bar = document.getElementById(id);
        if (bar) bar.style.width = Math.min(100, Math.max(0, pct)) + '%';
    };
    const setPct = (id, pct) => {
        const el = document.getElementById(id);
        if (el) el.textContent = Math.round(pct) + '%';
    };
    setBar('bar-prop', fuelPct);
    setPct('pct-prop', fuelPct);
    setBar('bar-ai', fusionPct);
    setPct('pct-ai', fusionPct);
}

function updateCopilotCards(data) {
    const cards = document.getElementById('copilot-cards');
    if (!cards) return;
    const s = data.spacecraft_status || {};
    const lines = [
        'Optimal trajectory confirmed. Fuel efficiency nominal. T+00:12',
        'Micro-debris field detected ahead. Recommending correction. T+00:34',
        'Fusion harmonics nominal. Thermal protection nominal. T+01:07',
    ];
    if (s.altitude != null) lines[0] = 'Altitude ' + s.altitude.toFixed(0) + ' km. Trajectory nominal. T+00:12';
    if (s.debris_remaining != null) lines[1] = s.debris_remaining + ' debris in range. Recommending correction. T+00:34';
    cards.innerHTML = lines.map((text, i) => {
        const cls = ['card-blue', 'card-orange', 'card-green'][i];
        return '<div class="copilot-card ' + cls + '">' + text + '</div>';
    }).join('');
}

function updateThreatCount(data) {
    const n = (data.debris_positions && data.debris_positions.length) || (data.spacecraft_status && data.spacecraft_status.debris_remaining) || 0;
    const el = document.getElementById('threat-count');
    if (el) el.textContent = n;
}

function updateBatteryShields(data) {
    const fuels = data.fuels || [];
    const conf = data.confidences || [];
    const fuelPct = fuels.length && fuels[0] ? (fuels[fuels.length - 1] / fuels[0]) * 100 : 89;
    const shieldPct = conf.length ? Math.round(conf[conf.length - 1] * 47 + 50) : 47;
    const batFill = document.getElementById('battery-fill');
    const batPct = document.getElementById('battery-pct');
    const shieldEl = document.getElementById('shields-pct');
    if (batFill) batFill.style.width = Math.min(100, fuelPct) + '%';
    if (batPct) batPct.textContent = Math.round(fuelPct) + '%';
    if (shieldEl) shieldEl.textContent = Math.min(100, shieldPct);
}

// Terminal / logs — append real-time style lines
function appendTerminalLine(text, className) {
    const out = document.getElementById('terminal-output');
    if (!out) return;
    const line = document.createElement('div');
    line.className = 'line' + (className ? ' ' + className : '');
    line.textContent = text;
    out.appendChild(line);
    out.scrollTop = out.scrollHeight;
}

function updateTerminal(data) {
    const steps = data.times ? data.times.length : 0;
    appendTerminalLine('[ORBIT] Track updated. Points: ' + steps, 'sys');
    appendTerminalLine('[TELEMETRY] ECI stream received.', 'sys');
    appendTerminalLine('[FUSION] Confidence nominal.', 'ok');
    if (data.spacecraft_status) {
        const s = data.spacecraft_status;
        appendTerminalLine('[RES] Altitude ' + (s.altitude != null ? s.altitude.toFixed(1) : '—') + ' km | Fuel ' + (s.fuel != null ? s.fuel.toFixed(1) : '—') + ' kg | Debris ' + (s.debris_remaining != null ? s.debris_remaining : '—'), 'line');
    }
    appendTerminalLine('[SYS] Mission dashboard ready.', 'ok');
}

function updateStatus(data) {
    const set = (id, text) => {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    };
    set('status-network', 'OK');
    set('status-compute', 'LOW');
    set('status-model', 'LOADED');
    set('status-runtime', 'NOMINAL');
    set('status-fusion', 'ACTIVE');
    if (data.spacecraft_status) {
        const s = data.spacecraft_status;
        if (s.altitude != null) set('status-runtime', 'ALT ' + s.altitude.toFixed(0) + ' km');
    }
}

// 3D Orbit visualization
function renderOrbitChart(data) {
    const positions = data.positions;
    const debrisPos = data.debris_positions;
    
    // Generate Earth sphere
    const earthTheta = Array.from({length: 20}, (_, i) => i * Math.PI / 19);
    const earthPhi = Array.from({length: 40}, (_, i) => i * 2 * Math.PI / 39);
    
    const earthX = [], earthY = [], earthZ = [];
    earthTheta.forEach(theta => {
        const row_x = [], row_y = [], row_z = [];
        earthPhi.forEach(phi => {
            row_x.push(EARTH_RADIUS * Math.sin(theta) * Math.cos(phi));
            row_y.push(EARTH_RADIUS * Math.sin(theta) * Math.sin(phi));
            row_z.push(EARTH_RADIUS * Math.cos(theta));
        });
        earthX.push(row_x);
        earthY.push(row_y);
        earthZ.push(row_z);
    });
    
    const traces = [
        // Earth
        {
            type: 'surface',
            x: earthX,
            y: earthY,
            z: earthZ,
            colorscale: [[0, '#1a3a52'], [1, '#1a3a52']],
            showscale: false,
            opacity: 0.5,
            hoverinfo: 'skip',
        },
        // Spacecraft trajectory
        {
            type: 'scatter3d',
            mode: 'lines',
            x: positions.map(p => p[0]),
            y: positions.map(p => p[1]),
            z: positions.map(p => p[2]),
            line: {
                color: COLORS.cyan,
                width: 3,
            },
            name: 'Trajectory',
        },
        // Start marker
        {
            type: 'scatter3d',
            mode: 'markers',
            x: [positions[0][0]],
            y: [positions[0][1]],
            z: [positions[0][2]],
            marker: {
                size: 6,
                color: COLORS.green,
                symbol: 'circle',
                line: { color: '#fff', width: 1 },
            },
            name: 'Start',
        },
        // Current position
        {
            type: 'scatter3d',
            mode: 'markers',
            x: [positions[positions.length - 1][0]],
            y: [positions[positions.length - 1][1]],
            z: [positions[positions.length - 1][2]],
            marker: {
                size: 8,
                color: COLORS.orange,
                symbol: 'diamond',
                line: { color: '#fff', width: 1 },
            },
            name: 'Current',
        },
    ];
    
    // Add debris (separate safe and dangerous)
    if (debrisPos.length > 0) {
        const scPos = positions.length ? positions[positions.length - 1] : [0, 0, 0];
        const safeDebris = [];
        const dangerousDebris = [];
        
        debrisPos.forEach(p => {
            const dist = Math.sqrt((p[0]-scPos[0])**2 + (p[1]-scPos[1])**2 + (p[2]-scPos[2])**2);
            if (dist < 0.1) {  // Less than 100m - dangerous
                dangerousDebris.push(p);
            } else {
                safeDebris.push(p);
            }
        });
        
        if (safeDebris.length > 0) {
            traces.push({
                type: 'scatter3d',
                mode: 'markers',
                x: safeDebris.map(p => p[0]),
                y: safeDebris.map(p => p[1]),
                z: safeDebris.map(p => p[2]),
                marker: {
                    size: 3,
                    color: COLORS.orange,
                    opacity: 0.6,
                },
                name: `Debris (${safeDebris.length})`,
            });
        }
        
        if (dangerousDebris.length > 0) {
            traces.push({
                type: 'scatter3d',
                mode: 'markers',
                x: dangerousDebris.map(p => p[0]),
                y: dangerousDebris.map(p => p[1]),
                z: dangerousDebris.map(p => p[2]),
                marker: {
                    size: 12,
                    color: '#ff4444',
                    opacity: 0.9,
                    symbol: 'x',
                    line: { color: '#fff', width: 2 },
                },
                name: `⚠️ ОПАСНЫЙ МУСОР (${dangerousDebris.length})`,
            });
        }
    }
    
    const layout = {
        ...commonLayout,
        scene: {
            xaxis: { 
                title: 'X (km)', 
                gridcolor: COLORS.grid,
                backgroundcolor: COLORS.panel,
                color: COLORS.textSecondary,
            },
            yaxis: { 
                title: 'Y (km)', 
                gridcolor: COLORS.grid,
                backgroundcolor: COLORS.panel,
                color: COLORS.textSecondary,
            },
            zaxis: { 
                title: 'Z (km)', 
                gridcolor: COLORS.grid,
                backgroundcolor: COLORS.panel,
                color: COLORS.textSecondary,
            },
            bgcolor: COLORS.panel,
            aspectmode: 'cube',
        },
        margin: { t: 10, r: 10, b: 10, l: 10 },
    };
    
    Plotly.newPlot('orbit-chart', traces, layout, {responsive: true});
}

// Position telemetry
function renderPositionChart(data) {
    const traces = [
        {
            x: data.times,
            y: data.positions.map(p => p[0]),
            name: 'X',
            type: 'scatter',
            mode: 'lines',
            line: { color: COLORS.blue, width: 2 },
        },
        {
            x: data.times,
            y: data.positions.map(p => p[1]),
            name: 'Y',
            type: 'scatter',
            mode: 'lines',
            line: { color: COLORS.cyan, width: 2 },
        },
        {
            x: data.times,
            y: data.positions.map(p => p[2]),
            name: 'Z',
            type: 'scatter',
            mode: 'lines',
            line: { color: COLORS.purple, width: 2 },
        },
    ];
    
    const layout = {
        ...commonLayout,
        xaxis: {
            title: 'Time (s)',
            gridcolor: COLORS.grid,
            color: COLORS.textSecondary,
        },
        yaxis: {
            title: 'Position (km)',
            gridcolor: COLORS.grid,
            color: COLORS.textSecondary,
        },
    };
    
    Plotly.newPlot('position-chart', traces, layout, {responsive: true});
}

// Velocity and fusion confidence
function renderVelocityChart(data) {
    const traces = [
        {
            x: data.times,
            y: data.speeds,
            name: 'Speed (km/s)',
            type: 'scatter',
            mode: 'lines',
            line: { color: COLORS.green, width: 2 },
            fill: 'tozeroy',
            fillcolor: 'rgba(63, 185, 80, 0.1)',
        },
        {
            x: data.times,
            y: data.confidences,
            name: 'Fusion Confidence',
            type: 'scatter',
            mode: 'lines',
            line: { color: COLORS.purple, width: 2, dash: 'dash' },
            yaxis: 'y2',
            fill: 'tozeroy',
            fillcolor: 'rgba(165, 165, 255, 0.1)',
        },
    ];
    
    const layout = {
        ...commonLayout,
        xaxis: {
            title: 'Time (s)',
            gridcolor: COLORS.grid,
            color: COLORS.textSecondary,
        },
        yaxis: {
            title: 'Speed (km/s)',
            gridcolor: COLORS.grid,
            color: COLORS.textSecondary,
        },
        yaxis2: {
            title: 'Confidence',
            overlaying: 'y',
            side: 'right',
            gridcolor: 'transparent',
            range: [0, 1.05],
            color: COLORS.purple,
        },
    };
    
    Plotly.newPlot('velocity-chart', traces, layout, {responsive: true});
}

// Resources (fuel and debris count) with danger overlay
function renderResourcesChart(data) {
    const traces = [
        {
            x: data.times,
            y: data.fuels,
            name: 'Fuel (kg)',
            type: 'scatter',
            mode: 'lines',
            line: { color: COLORS.orange, width: 2 },
            fill: 'tozeroy',
            fillcolor: 'rgba(255, 140, 66, 0.1)',
        },
        {
            x: data.times,
            y: data.debris_counts,
            name: 'Debris Count',
            type: 'scatter',
            mode: 'lines',
            line: { color: COLORS.cyan, width: 2, dash: 'dot' },
            yaxis: 'y2',
            fill: 'tozeroy',
            fillcolor: 'rgba(0, 212, 255, 0.1)',
        },
    ];
    
    // Add danger level if available
    if (data.danger_levels && data.danger_levels.length > 0) {
        const dangerColor = data.danger_levels[data.danger_levels.length - 1] > 0.7 ? '#ff4444' :
                           data.danger_levels[data.danger_levels.length - 1] > 0.4 ? '#ff8c42' :
                           data.danger_levels[data.danger_levels.length - 1] > 0.1 ? '#ffaa00' : COLORS.green;
        
        traces.push({
            x: data.times,
            y: data.danger_levels,
            name: 'Уровень опасности',
            type: 'scatter',
            mode: 'lines',
            line: { color: dangerColor, width: 2.5 },
            yaxis: 'y3',
            fill: 'tozeroy',
            fillcolor: dangerColor.replace(')', ', 0.2)').replace('rgb', 'rgba'),
        });
    }
    
    const layout = {
        ...commonLayout,
        xaxis: {
            title: 'Time (s)',
            gridcolor: COLORS.grid,
            color: COLORS.textSecondary,
        },
        yaxis: {
            title: 'Fuel (kg)',
            gridcolor: COLORS.grid,
            color: COLORS.textSecondary,
        },
        yaxis2: {
            title: 'Debris Count',
            overlaying: 'y',
            side: 'right',
            gridcolor: 'transparent',
            color: COLORS.cyan,
        },
        yaxis3: {
            title: 'Уровень опасности',
            overlaying: 'y',
            side: 'right',
            position: 0.95,
            gridcolor: 'transparent',
            range: [0, 1.05],
            color: data.danger_levels && data.danger_levels.length > 0 && data.danger_levels[data.danger_levels.length - 1] > 0.1 ? '#ff4444' : COLORS.textSecondary,
        },
    };
    
    Plotly.newPlot('resources-chart', traces, layout, {responsive: true});
}

// Update danger status
function updateDangerStatus(data) {
    const dangerLevels = data.danger_levels || [];
    const currentDanger = dangerLevels.length > 0 ? dangerLevels[dangerLevels.length - 1] : 0.0;
    
    // Update danger gauge
    const dangerBar = document.getElementById('danger-level-bar');
    const dangerValue = document.getElementById('danger-level-value');
    if (dangerBar) {
        dangerBar.style.width = (currentDanger * 100) + '%';
        if (currentDanger > 0.7) {
            dangerBar.style.background = '#ff4444';
        } else if (currentDanger > 0.4) {
            dangerBar.style.background = '#ff8c42';
        } else if (currentDanger > 0.1) {
            dangerBar.style.background = '#ffaa00';
        } else {
            dangerBar.style.background = COLORS.green;
        }
    }
    if (dangerValue) {
        dangerValue.textContent = (currentDanger * 100).toFixed(0) + '%';
        dangerValue.style.color = currentDanger > 0.7 ? '#ff4444' :
                                  currentDanger > 0.4 ? '#ff8c42' :
                                  currentDanger > 0.1 ? '#ffaa00' : COLORS.green;
    }
    
    // Update stats
    const collisionCount = document.getElementById('collision-count');
    const anomalyCount = document.getElementById('anomaly-count');
    const lowFuelCount = document.getElementById('low-fuel-count');
    if (collisionCount) collisionCount.textContent = (data.collision_warnings || []).length;
    if (anomalyCount) anomalyCount.textContent = (data.anomaly_detections || []).length;
    if (lowFuelCount) lowFuelCount.textContent = (data.low_fuel_warnings || []).length;
    
    // Update footer status
    const sysStatus = document.getElementById('sys-status');
    const statusDot = document.getElementById('status-dot');
    const warningsCount = document.getElementById('warnings-count');
    const warningsText = document.getElementById('warnings-text');
    const warningsNum = document.getElementById('warnings-num');
    
    const totalWarnings = (data.collision_warnings || []).length + 
                         (data.anomaly_detections || []).length + 
                         (data.low_fuel_warnings || []).length;
    
    let statusText = 'SYS NOMINAL';
    let statusColor = COLORS.green;
    
    if (currentDanger > 0.7) {
        statusText = 'STATUS: CRITICAL';
        statusColor = '#ff4444';
    } else if (currentDanger > 0.4) {
        statusText = 'STATUS: WARNING';
        statusColor = '#ff8c42';
    } else if (currentDanger > 0.1) {
        statusText = 'STATUS: CAUTION';
        statusColor = '#ffaa00';
    }
    
    if (sysStatus) {
        sysStatus.textContent = statusText;
        sysStatus.style.color = statusColor;
    }
    
    if (statusDot) {
        statusDot.style.background = statusColor;
        statusDot.style.boxShadow = '0 0 10px ' + statusColor;
    }
    
    if (totalWarnings > 0) {
        if (warningsCount) warningsCount.style.display = 'inline';
        if (warningsText) warningsText.style.display = 'inline';
        if (warningsNum) warningsNum.textContent = totalWarnings;
    } else {
        if (warningsCount) warningsCount.style.display = 'none';
        if (warningsText) warningsText.style.display = 'none';
    }
}

// Update danger warnings panel
function updateDangerWarnings(data) {
    const dangerPanel = document.getElementById('danger-panel');
    const dangerContent = document.getElementById('danger-content');
    
    if (!dangerPanel || !dangerContent) return;
    
    const warnings = [];
    
    // Collision warnings
    (data.collision_warnings || []).forEach(w => {
        warnings.push({
            icon: '⚠️',
            text: `СТОЛКНОВЕНИЕ! Вероятность: ${(w.probability * 100).toFixed(1)}%`,
            severity: w.severity,
            time: w.time,
        });
    });
    
    // Anomaly detections
    (data.anomaly_detections || []).forEach(a => {
        warnings.push({
            icon: '🚨',
            text: `АНОМАЛИЯ! Уровень: ${(a.score * 100).toFixed(1)}%`,
            severity: a.score > 0.7 ? 'HIGH' : 'MEDIUM',
            time: a.time,
        });
    });
    
    // Low fuel warnings
    (data.low_fuel_warnings || []).forEach(f => {
        warnings.push({
            icon: '⛽',
            text: `НИЗКИЙ УРОВЕНЬ ТОПЛИВА! ${(f.fuel_ratio * 100).toFixed(1)}%`,
            severity: 'MEDIUM',
            time: f.time,
        });
    });
    
    if (warnings.length > 0) {
        dangerPanel.style.display = 'block';
        dangerContent.innerHTML = warnings.map(w => {
            const severityClass = w.severity === 'HIGH' ? 'danger-high' : 'danger-medium';
            return `<div class="danger-warning ${severityClass}">
                <span class="danger-warning-icon">${w.icon}</span>
                <span class="danger-warning-text">${w.text}</span>
                <span class="danger-warning-time">T+${w.time.toFixed(1)}s</span>
            </div>`;
        }).join('');
    } else {
        dangerPanel.style.display = 'none';
    }
}

// Initialize on page load: cover visible, dashboard hidden; no auto-load.
function initCover() {
    startLiveTime();

    var cover = document.getElementById('cover');
    var dashboard = document.getElementById('dashboard');
    var startBtn = document.getElementById('cover-start-btn');
    var coverStatus = document.getElementById('cover-status');
    var seedBlock = document.getElementById('cover-seed-block');
    var generatedSeedEl = document.getElementById('generated-seed-value');
    var seedInput = document.getElementById('seed-input');

    function setCoverStatus(text, isError) {
        if (!coverStatus) return;
        coverStatus.textContent = text || '';
        coverStatus.classList.toggle('error', !!isError);
    }

    function setLoading(loading) {
        if (startBtn) {
            startBtn.disabled = loading;
            startBtn.textContent = loading ? 'Generating…' : 'Start';
        }
        if (!loading) setCoverStatus('');
    }

    function setAddLoading(loading) {
        var addBtn = document.getElementById('cover-add-btn');
        if (addBtn) {
            addBtn.disabled = loading;
            addBtn.textContent = loading ? 'Loading…' : 'Add';
        }
        if (!loading) setCoverStatus('');
    }

    function showSeedOnCover(seed) {
        if (generatedSeedEl) generatedSeedEl.textContent = seed;
        if (seedBlock) seedBlock.style.display = 'block';
        if (seedInput) seedInput.value = String(seed);
    }

    // Direct listener on Seed button so it always works (some environments miss delegation)
    var seedBtn = document.getElementById('cover-seed-btn');
    if (seedBtn) {
        seedBtn.addEventListener('click', function(ev) {
            ev.preventDefault();
            ev.stopPropagation();
            var seed = generateSeed();
            showSeedOnCover(seed);
            setCoverStatus('Сид сгенерирован. Нажмите Start для запуска сайта с этим сидом.');
        });
    }

    // One listener on cover: delegate clicks; use closest() so click on button text still works
    if (cover) {
        cover.addEventListener('click', function(ev) {
            var target = ev.target;
            if (!target || !target.closest) return;
            var clicked = target.closest('#cover-seed-btn') || target.closest('#cover-seed-copy') || target.closest('#cover-add-btn') || target.closest('#cover-start-btn');
            if (!clicked) return;

            if (clicked.id === 'cover-seed-btn') {
                var seed = generateSeed();
                showSeedOnCover(seed);
                setCoverStatus('Сид сгенерирован. Нажмите Start для запуска сайта с этим сидом.');
                return;
            }

            if (clicked.id === 'cover-seed-copy') {
                var value = generatedSeedEl ? generatedSeedEl.textContent : '';
                if (!value) return;
                if (navigator.clipboard && navigator.clipboard.writeText) {
                    navigator.clipboard.writeText(value).then(function() {
                        setCoverStatus('Сид скопирован.', false);
                    }).catch(function() {
                        setCoverStatus('Не удалось скопировать.', true);
                    });
                } else {
                    setCoverStatus('Копирование недоступно в этом браузере.', true);
                }
                return;
            }

            if (clicked.id === 'cover-add-btn') {
                var raw = (seedInput && (seedInput.value || '').trim()) || '';
                var seedAdd = parseInt(raw, 10);
                if (raw === '' || isNaN(seedAdd)) {
                    setCoverStatus('Введите сид (целое число).', true);
                    return;
                }
                setAddLoading(true);
                setCoverStatus('Loading mission data…');
                loadMissionData(seedAdd)
                    .then(function() {
                        if (cover) cover.style.display = 'none';
                        if (dashboard) {
                            dashboard.classList.remove('dashboard-hidden');
                            dashboard.classList.add('dashboard-visible');
                        }
                    })
                    .catch(function(err) {
                        console.error('Error loading mission data:', err);
                        setCoverStatus('Failed to load mission data. Try again.', true);
                    })
                    .finally(setAddLoading);
                return;
            }

            if (clicked.id === 'cover-start-btn') {
                var rawStart = (seedInput && (seedInput.value || '').trim()) || '';
                var seedStart = rawStart === '' ? null : parseInt(rawStart, 10);
                if (rawStart !== '' && isNaN(seedStart)) {
                    setCoverStatus('Введите сид (целое число) или нажмите Seed и затем Start.', true);
                    return;
                }
                setLoading(true);
                setCoverStatus('Запуск с сидом ' + (seedStart != null ? seedStart : 'по умолчанию') + '…');
                loadMissionData(seedStart)
                    .then(function() {
                        if (cover) cover.style.display = 'none';
                        if (dashboard) {
                            dashboard.classList.remove('dashboard-hidden');
                            dashboard.classList.add('dashboard-visible');
                        }
                    })
                    .catch(function(err) {
                        console.error('Error loading mission data:', err);
                        setCoverStatus('Failed to load mission data. Try again.', true);
                    })
                    .finally(setLoading);
            }
        });
    }

    // Redraw radar on window resize
    var resizeTimeout;
    window.addEventListener('resize', function() {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(function() {
            if (lastMissionData) drawRadar(lastMissionData);
        }, 250);
    });
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initCover);
} else {
    initCover();
}

